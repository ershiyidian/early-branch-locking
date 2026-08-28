#!/usr/bin/env python3
"""Compute semantic-first-calculation entropy from resolved semantic_boundary_analysis spans."""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from early_branch_locking.core.countdown_shared import canonicalize_expression
def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/rlvr/outputs/experiments/e4_semantic_boundary_v1');p.add_argument('--tag',default='v1');p.add_argument('--traces-per-origin',type=int,default=2000);p.add_argument('--draws',type=int,default=10000);p.add_argument('--seed',type=int,default=1729);return p.parse_args(argv)
def label(t):
 t=' '.join(str(t).split());left=t.split('=',1)[0].strip() if '=' in t else t;c,_=canonicalize_expression(left);return c or t
def entropy(c):
 """Miller-Madow corrected entropy of the sampled semantic labels."""
 n=sum(c.values())
 if not n:return np.nan
 plugin=-sum(v/n*np.log(v/n) for v in c.values() if v)
 return float(plugin+(len(c)-1)/(2*n))
def sampled_entropy(frame, rng, n, draws):
 """Sample traces without replacement, then cluster-bootstrap problems."""
 if not len(frame): return np.nan,np.nan,np.nan,0,0
 sampled=frame.iloc[rng.permutation(len(frame))[:min(n,len(frame))]].copy()
 by_problem=sampled.groupby('problem_id').semantic_label.agg(list)
 labels=list(by_problem)
 observed=entropy(Counter(label for values in labels for label in values))
 bootstrap_values=[]
 for indices in rng.integers(len(labels),size=(draws,len(labels))):
  bootstrap_values.append(entropy(Counter(label for i in indices for label in labels[int(i)])))
 return observed,float(np.quantile(bootstrap_values,.025)),float(np.quantile(bootstrap_values,.975)),len(sampled),len(labels)
def main(argv=None):
 a=parse(argv);ann=pd.read_json(a.root/'semantic_boundary_annotations_v1.jsonl',lines=True);idx=pd.read_json(a.root/'index_private.jsonl',lines=True);df=ann.merge(idx,on=['item_id','dup_index'],validate='one_to_one');df=df[(df.dup_index==0)&df.resolver_status.eq('resolved')&df.benchmark.isin(['gsm8k','math500'])].copy();df['semantic_label']=df.resolved_span_text.map(label);rows=[];rng=np.random.default_rng(a.seed)
 for (bench,origin),g in df.groupby(['benchmark','origin']):
  m,l,h,n_traces,n_problems=sampled_entropy(g,rng,a.traces_per_origin,a.draws);rows.append({'benchmark':bench,'origin':origin,'semantic_first_calc_entropy':m,'ci_lo':l,'ci_hi':h,'n_problems':n_problems,'n_traces':n_traces,'population_resolved_traces':len(g),'traces_requested_per_origin':a.traces_per_origin,'bootstrap_draws':a.draws,'bootstrap_seed':a.seed,'statistical_unit':'problem_cluster'})
 out=a.root/f'e4_semantic_entropy_{a.tag}.csv';
 if out.exists():raise FileExistsError(out)
 pd.DataFrame(rows).to_csv(out,index=False);print(out)
if __name__=='__main__':main()
