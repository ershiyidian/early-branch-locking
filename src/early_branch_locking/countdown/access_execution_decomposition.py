#!/usr/bin/env python3
"""Join splice logit access with sampled conditional execution.

Hypothesis: alternative families can retain conditional execution probability
even when their entrance share is very small. Inputs: prefix_recovery logit-access and
per-instance splice tables. Outputs: aligned family-level table and summary.
the paired step-50 sampler is a separate GPU follow-up.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];MET=ROOT/'data/analysis_results/rlvr_passk/metrics'
def argspec(v=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--logit',type=Path,default=MET/'prefix_recovery_logit_access_splice_v1.csv');p.add_argument('--per-instance',type=Path,default=MET/'prefix_recovery_per_instance_splice_v1.csv');p.add_argument('--paired-logit50',type=Path);p.add_argument('--paired-logit275',type=Path);p.add_argument('--paired-per50',type=Path);p.add_argument('--paired-per275',type=Path);p.add_argument('--out-dir',type=Path,default=MET);p.add_argument('--tag',default='v1');return p.parse_args(v)
KEYS=['problem_index','scaffold_sample_index','branch']


def _gini(values: np.ndarray) -> float:
    """Return the finite-sample Gini coefficient of a probability vector."""
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0 or float(x.sum()) <= 0:
        return float("nan")
    n = len(x)
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum()))


def _menu_access_stats(path: Path, label: str) -> pd.DataFrame:
    """Collapse family shares to one concentration row per menu."""
    frame = pd.read_csv(path)
    frame.branch = frame.branch.astype(str)
    rows = []
    for (pid, scaffold), group in frame.groupby(['problem_index', 'scaffold_sample_index'], sort=True):
        shares = group['access_logit_share_within_menu'].to_numpy(float)
        shares = shares[np.isfinite(shares) & (shares >= 0)]
        if not len(shares) or float(shares.sum()) <= 0:
            continue
        shares = shares / shares.sum()
        entropy = float(-np.sum(shares * np.log(np.maximum(shares, 1e-300))))
        rows.append({
            'problem_index': int(pid),
            'scaffold_sample_index': int(scaffold),
            f'{label}_menu_entropy': entropy,
            f'{label}_menu_entropy_norm': entropy / np.log(len(shares)) if len(shares) > 1 else 0.0,
            f'{label}_menu_top1_share': float(np.max(shares)),
            f'{label}_menu_gini': _gini(shares),
            f'{label}_menu_n_families': int(len(shares)),
        })
    return pd.DataFrame(rows)


def _bootstrap(values, rng, draws=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float('nan'), float('nan'), float('nan')
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)])
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main(a):
 if all(getattr(a,k) is not None for k in ('paired_logit50','paired_logit275','paired_per50','paired_per275')):
  paired(a); return
 logit=pd.read_csv(a.logit); per=pd.read_csv(a.per_instance);per=per[per.arm.eq('feasible')].copy();per.branch=per.branch.astype(str);logit.branch=logit.branch.astype(str)
 keys=['problem_index','scaffold_sample_index','branch'];x=logit.merge(per[keys+['in_family_valid_rate','any_valid_rate','chance','n_solutions_in_family','original_overall_ok']],on=keys,how='left',validate='one_to_one')
 x['access_lt_005']=x.access_logit_share_within_menu<.05;x['execution_over_chance']=x.in_family_valid_rate-x.chance
 summary=x.groupby('access_lt_005',as_index=False).agg(median_execution=('in_family_valid_rate','median'),mean_execution=('in_family_valid_rate','mean'),mean_chance=('chance','mean'),n=('branch','size'),join_coverage=('in_family_valid_rate',lambda z:z.notna().mean()))
 a.out_dir.mkdir(parents=True,exist_ok=True);x.to_csv(a.out_dir/f'prefix_recovery_access_execution_per_instance_{a.tag}.csv',index=False);summary.to_csv(a.out_dir/f'prefix_recovery_access_execution_summary_{a.tag}.csv',index=False);print(summary.to_string(index=False))

def paired(a):
 """Join identical feasible instances at both checkpoints and bootstrap deltas."""
 frames=[]
 for path, label, cols in ((a.paired_logit50,'access50',['access_logit_share_within_menu']),
                           (a.paired_logit275,'access275',['access_logit_share_within_menu'])):
  x=pd.read_csv(path); x.branch=x.branch.astype(str)
  frames.append(x[KEYS+cols].rename(columns={cols[0]:label}))
 for path, label in ((a.paired_per50,'exec50'),(a.paired_per275,'exec275')):
  x=pd.read_csv(path); x=x[x.arm.eq('feasible')].copy(); x.branch=x.branch.astype(str)
  frames.append(x[KEYS+['in_family_valid_rate','any_valid_rate','chance']].rename(columns={
   'in_family_valid_rate':f'{label}_in_family','any_valid_rate':f'{label}_any','chance':f'{label}_chance'}))
 joined=reduce(lambda left,right:left.merge(right,on=KEYS,how='outer',validate='one_to_one'),frames)
 if joined.empty or joined.isna().any().any(): raise ValueError('paired G3 join is incomplete or duplicated')
 # Concentration is defined at the menu level.  A raw share delta sums to
 # zero within every menu, so report shape changes (entropy/top-1/Gini) and
 # the mean absolute per-family share change instead.
 for path, label in ((a.paired_logit50, 'access50'), (a.paired_logit275, 'access275')):
  stats = _menu_access_stats(path, label)
  if stats.empty:
   raise ValueError(f'no menu concentration rows in {path}')
  joined = joined.merge(stats, on=['problem_index', 'scaffold_sample_index'],
                        how='left', validate='many_to_one')
 if joined[[c for c in joined.columns if c.startswith('access50_menu_') or c.startswith('access275_menu_')]].isna().any().any():
  raise ValueError('paired G3 menu concentration join is incomplete')
 joined['delta_access']=joined.access275-joined.access50
 joined['delta_execution']=joined.exec275_in_family-joined.exec50_in_family
 joined['abs_delta_access']=joined.delta_access.abs()
 for metric in ('menu_entropy', 'menu_entropy_norm', 'menu_top1_share', 'menu_gini'):
  joined[f'delta_{metric}'] = joined[f'access275_{metric}'] - joined[f'access50_{metric}']
 joined['access275_lt_005']=joined.access275 < .05
 joined['access50_lt_005']=joined.access50 < .05
 rng=np.random.default_rng(1729)
 def ci(values):
  values=np.asarray(values,float); means=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(10000)])
  return float(values.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))
 rows=[]
 for low, group in joined.groupby('access275_lt_005',sort=True):
  rows.append({'metric':'execution_in_family','subset':f'access275_lt_005={low}','mean':float(group.exec275_in_family.mean()),'median':float(group.exec275_in_family.median()),'mean_chance':float(group.exec275_chance.mean()),'n':len(group),'n_problems':group.problem_index.nunique()})
 problem=joined.groupby('problem_index')[['delta_access','delta_execution']].mean()
 for col in problem.columns:
  mean,lo,hi=ci(problem[col].dropna().to_numpy()); rows.append({'metric':col,'subset':'problem_paired','mean':mean,'ci_lo':lo,'ci_hi':hi,'n':len(problem),'n_problems':len(problem)})
 corr=float(problem.delta_access.corr(problem.delta_execution))
 rows.append({'metric':'problem_corr_delta_access_execution','subset':'problem_paired','mean':corr,'n':len(problem),'n_problems':len(problem)})
 rng=np.random.default_rng(1729)
 # One row per menu avoids weighting problems with many entrance families.
 menu = joined.drop_duplicates(['problem_index', 'scaffold_sample_index']).copy()
 for metric in ('menu_entropy', 'menu_entropy_norm', 'menu_top1_share', 'menu_gini'):
  values = menu[f'delta_{metric}'].to_numpy(float)
  mean, lo, hi = _bootstrap(values, rng)
  rows.append({'metric':f'delta_{metric}', 'subset':'menu_paired', 'mean':mean,
               'ci_lo':lo, 'ci_hi':hi, 'n':len(values),
               'n_problems':menu.problem_index.nunique()})
 values = joined['abs_delta_access'].to_numpy(float)
 mean, lo, hi = _bootstrap(values, rng)
 rows.append({'metric':'mean_abs_family_access_delta', 'subset':'family_paired',
              'mean':mean, 'ci_lo':lo, 'ci_hi':hi, 'n':len(values),
              'n_problems':joined.problem_index.nunique()})
 a.out_dir.mkdir(parents=True,exist_ok=True)
 joined.to_csv(a.out_dir/f'prefix_recovery_access_execution_paired_{a.tag}.csv',index=False)
 pd.DataFrame(rows).to_csv(a.out_dir/f'prefix_recovery_access_execution_paired_summary_{a.tag}.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main(argspec())
