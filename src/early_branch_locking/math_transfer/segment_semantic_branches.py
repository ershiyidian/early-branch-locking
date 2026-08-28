#!/usr/bin/env python3
"""CPU resegmentation of semantic_boundary_analysis per-token caches at parser and semantic boundaries."""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from early_branch_locking.math_transfer.analyze_segment_suppression import token_boundary
def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/rlvr/outputs/experiments/e4_semantic_boundary_v1');p.add_argument('--model-root',type=Path,default=ROOT/'model/math_base_7b');p.add_argument('--tag',default='v1');p.add_argument('--rdd-bandwidth',type=int,default=20);p.add_argument('--draws',type=int,default=10000);p.add_argument('--seed',type=int,default=0);p.add_argument('--worst-case',action='store_true');p.add_argument('--rdd-only',action='store_true',help='write only the RDD/placebo artifacts; requires an existing primary resegmentation');return p.parse_args(argv)
def boot(x,d,s):
 x=np.asarray(x,float);r=np.random.default_rng(s);b=x[r.integers(len(x),size=(d,len(x)))].mean(1);return float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))
def rate(nll,b):
 b=min(max(int(b),0),len(nll));return (float(nll[:b].mean()) if b else np.nan,float(nll[b:].mean()) if b<len(nll) else np.nan)
def did_at(nll_base,nll_rl,b):
 bc,be=rate(nll_base,b);rc,re=rate(nll_rl,b)
 return (bc-rc)-(be-re),bc,be,rc,re
def validate_output_targets(out,detail,worst,worst_case):
 if worst_case:
  if not out.exists() or not detail.exists():raise RuntimeError('worst-case sensitivity requires completed primary resegmentation outputs')
  if worst.exists():raise FileExistsError(worst)
 else:
  if out.exists() or detail.exists():raise FileExistsError(out)
def summary_rows(per,draws,seed):
 summary=[]
 for (bench,boundary,origin),g in per.groupby(['benchmark','boundary','origin']):
  vals=g.groupby('problem_id').did.mean().to_numpy(float);m,l,h=boot(vals,draws,seed)
  summary.append({'benchmark':bench,'boundary':boundary,'origin':origin,'metric':'early_minus_execution_shift','mean':m,'ci_lo':l,'ci_hi':h,'n_problems':len(vals),'n_traces':len(g),'bootstrap_draws':draws,'bootstrap_seed':seed,'statistical_unit':'problem_cluster'})
 return summary
def primary_resegment_items(all_items):
 """Keep only exact semantic annotations with both required character cuts."""
 return all_items[all_items.resolver_status.eq('resolved')&all_items.parser_char_end.notna()&all_items.semantic_char_start.notna()].copy()


CALCULATION_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d+)?)\s*(?:[+\-*/xX\u00d7\u00f7])\s*[-+]?(?:\d+(?:\.\d+)?)"
)
SENTENCE_BREAK_RE = re.compile(r"[.!?;\n](?:\s|$)")


def local_linear_jump(base_nll, rl_nll, boundary, bandwidth):
 """Estimate the post-boundary jump in base-minus-RL token NLL.

 The model is ``y = a + tau * post + b*t + c*post*t`` over the symmetric
 token window.  ``tau`` is the RDD estimand at the first calculation boundary.
 """
 if len(base_nll)!=len(rl_nll): raise ValueError('token-cache length mismatch')
 b=int(boundary); w=int(bandwidth)
 left=max(0,b-w);right=min(len(base_nll),b+w)
 positions=np.arange(left,right)-b
 if len(positions)<4 or not np.any(positions<0) or not np.any(positions>=0): return np.nan
 post=(positions>=0).astype(float)
 design=np.c_[np.ones(len(positions)),post,positions,post*positions]
 values=np.asarray(base_nll[left:right],float)-np.asarray(rl_nll[left:right],float)
 if not np.isfinite(values).all(): return np.nan
 return float(np.linalg.lstsq(design,values,rcond=None)[0][1])


def nearest_sentence_boundary(response, true_char, tokenizer):
 candidates=[match.end() for match in SENTENCE_BREAK_RE.finditer(str(response))]
 if not candidates: return None
 char=min(candidates,key=lambda value: abs(value-int(true_char)))
 return token_boundary(tokenizer,response,char)


def second_calculation_boundary(response, tokenizer):
 matches=list(CALCULATION_RE.finditer(str(response)))
 return token_boundary(tokenizer,response,matches[1].end()) if len(matches)>=2 else None


def cluster_summary(frame, value, draws, seed):
 rows=[]
 for (benchmark,origin),group in frame.groupby(['benchmark','origin'],sort=True):
  problem_values=group.groupby('problem_id')[value].mean().dropna().to_numpy(float)
  if not len(problem_values): continue
  mean,lo,hi=boot(problem_values,draws,seed)
  rows.append({'benchmark':benchmark,'origin':origin,'metric':value,'mean':mean,'ci_lo':lo,'ci_hi':hi,'n_problems':len(problem_values),'n_traces':len(group),'bootstrap_draws':draws,'bootstrap_seed':seed,'statistical_unit':'problem_cluster'})
 return pd.DataFrame(rows)


def rdd_and_placebos(items, tokenizer, cache_dir, bandwidth, draws, seed):
 """Compute RDD jumps and matched boundary controls from immutable semantic_boundary_analysis caches."""
 meta=[]
 for row in items.itertuples():
  base_path=cache_dir/f'{row.item_id}__math_base_7b.npz';rl_path=cache_dir/f'{row.item_id}__math_simple_rl_7b.npz'
  if not base_path.exists() or not rl_path.exists(): raise FileNotFoundError(f'missing cache for {row.item_id}')
  base=np.load(base_path)['nll'];rl=np.load(rl_path)['nll']
  if len(base)!=len(rl): raise ValueError(f'token-cache length mismatch for {row.item_id}')
  true=token_boundary(tokenizer,row.response,int(row.semantic_char_start))
  if true is None or not 0<int(true)<len(base): continue
  meta.append({'item_id':row.item_id,'benchmark':row.benchmark,'origin':row.origin,'problem_id':row.problem_id,'response':row.response,'semantic_char_start':int(row.semantic_char_start),'length':len(base),'true_boundary':int(true),'true_percentile':float(true/len(base))})
 if not meta: raise RuntimeError('no valid semantic boundaries for RDD/placebo')
 frame=pd.DataFrame(meta);rng=np.random.default_rng(seed);frame['placebo_position']=np.nan
 for _,indices in frame.groupby('benchmark',sort=True).groups.items():
  values=frame.loc[indices,'true_percentile'].to_numpy(float)
  frame.loc[indices,'placebo_position']=rng.permutation(values)
 records=[]
 for row in frame.itertuples():
  base=np.load(cache_dir/f'{row.item_id}__math_base_7b.npz')['nll'].astype(float)
  rl=np.load(cache_dir/f'{row.item_id}__math_simple_rl_7b.npz')['nll'].astype(float)
  positions={'true':row.true_boundary,'position_permuted':int(np.clip(row.placebo_position*row.length,1,row.length-1)),
             'sentence':nearest_sentence_boundary(row.response,row.semantic_char_start,tokenizer),
             'second_calculation':second_calculation_boundary(row.response,tokenizer)}
  for kind,boundary in positions.items():
   if boundary is None or not 0<int(boundary)<row.length: continue
   records.append({'item_id':row.item_id,'benchmark':row.benchmark,'origin':row.origin,'problem_id':row.problem_id,'boundary_kind':kind,'boundary_token':int(boundary),'rdd_jump':local_linear_jump(base,rl,boundary,bandwidth)})
 detail=pd.DataFrame(records)
 return detail,cluster_summary(detail,'rdd_jump',draws,seed)
def main(argv=None):
 a=parse(argv);from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(a.model_root),local_files_only=True,trust_remote_code=True,use_fast=False);ann=pd.read_json(a.root/'semantic_boundary_annotations_v1.jsonl',lines=True);idx=pd.read_json(a.root/'index_private.jsonl',lines=True);packet=pd.read_json(a.root/'packet_blind.jsonl',lines=True);all_items=ann.merge(idx,on=['item_id','dup_index'],validate='one_to_one').merge(packet[['item_id','dup_index','response']],on=['item_id','dup_index'],validate='one_to_one');all_items=all_items[all_items.dup_index==0].copy();df=primary_resegment_items(all_items);rows=[];worst_rows=[];missing=[]
 if a.rdd_only:
  primary=a.root/f'e4_resegment_rows_{a.tag}.parquet';rdd_path=a.root/f'e4_rdd_placebo_rows_{a.tag}.parquet';rdd_summary_path=a.root/f'e4_rdd_placebo_summary_{a.tag}.csv'
  if not primary.exists(): raise FileNotFoundError(f'rdd-only requires primary resegmentation: {primary}')
  if rdd_path.exists() or rdd_summary_path.exists(): raise FileExistsError(rdd_path)
  rdd_detail,rdd_summary=rdd_and_placebos(df,tok,a.root/'e4_token_cache',a.rdd_bandwidth,a.draws,a.seed)
  rdd_detail.to_parquet(rdd_path,index=False);rdd_summary.to_csv(rdd_summary_path,index=False)
  print(json.dumps({'rdd_rows':len(rdd_detail),'rdd_summary':str(rdd_summary_path)},sort_keys=True));return
 for r in df.itertuples():
  p=a.root/'e4_token_cache'/f'{r.item_id}__math_base_7b.npz';q=a.root/'e4_token_cache'/f'{r.item_id}__math_simple_rl_7b.npz'
  if not p.exists() or not q.exists():missing.append(r.item_id);continue
  base=np.load(p)['nll'].astype(float);rl=np.load(q)['nll'].astype(float)
  if len(base)!=len(rl):raise RuntimeError(f'token-cache length mismatch for {r.item_id}: base={len(base)} rl={len(rl)}')
  boundaries={'parser':token_boundary(tok,r.response,int(r.parser_char_end)),'semantic':token_boundary(tok,r.response,int(r.semantic_char_start))}
  for name,b in boundaries.items():
   if b is None:continue
   _,bc,be,rc,re=did_at(base,rl,b)
   if not all(np.isfinite([bc,be,rc,re])):continue
   rows.append({'item_id':r.item_id,'benchmark':r.benchmark,'origin':r.origin,'problem_id':r.problem_id,'boundary':name,'base_pre_nll':bc,'base_post_nll':be,'rl_pre_nll':rc,'rl_post_nll':re,'early_shift':bc-rc,'execution_shift':be-re,'did':(bc-rc)-(be-re),'boundary_token':int(b)})
 if missing:raise RuntimeError(f'missing token cache for {len(set(missing))} resolved items; run e4_token_cache first')
 # The sensitivity is intentionally separate from the semantic main result.  Each
 # unresolved annotation receives the token boundary with the most negative DiD,
 # i.e. the assignment least favorable to the preregistered positive-direction claim.
 if a.worst_case:
  for r in all_items[~all_items.resolver_status.eq('resolved')].itertuples():
   p=a.root/'e4_token_cache'/f'{r.item_id}__math_base_7b.npz';q=a.root/'e4_token_cache'/f'{r.item_id}__math_simple_rl_7b.npz'
   if not p.exists() or not q.exists():missing.append(r.item_id);continue
   base=np.load(p)['nll'].astype(float);rl=np.load(q)['nll'].astype(float)
   if len(base)!=len(rl):raise RuntimeError(f'token-cache length mismatch for {r.item_id}: base={len(base)} rl={len(rl)}')
   candidates=[(did_at(base,rl,b)[0],b) for b in range(1,len(base))]
   candidates=[item for item in candidates if np.isfinite(item[0])]
   if not candidates:continue
   _,b=min(candidates);d,bc,be,rc,re=did_at(base,rl,b)
   worst_rows.append({'item_id':r.item_id,'benchmark':r.benchmark,'origin':r.origin,'problem_id':r.problem_id,'boundary':'semantic_worst_case','base_pre_nll':bc,'base_post_nll':be,'rl_pre_nll':rc,'rl_post_nll':re,'early_shift':bc-rc,'execution_shift':be-re,'did':d,'boundary_token':int(b),'resolver_status':r.resolver_status,'assignment':'minimum_valid_token_DiD'})
 if missing:raise RuntimeError(f'missing token cache for {len(set(missing))} items; run e4_token_cache first')
 per=pd.DataFrame(rows);summary=summary_rows(per,a.draws,a.seed)
 out=a.root/f'e4_resegment_did_{a.tag}.csv';detail=a.root/f'e4_resegment_rows_{a.tag}.parquet';worst=a.root/f'e4_worst_case_{a.tag}.csv'
 validate_output_targets(out,detail,worst,a.worst_case)
 if a.worst_case:
  pd.DataFrame(summary_rows(pd.DataFrame(worst_rows),a.draws,a.seed)).to_csv(worst,index=False)
 else:
  per.to_parquet(detail,index=False);pd.DataFrame(summary).to_csv(out,index=False)
  rdd_detail,rdd_summary=rdd_and_placebos(df,tok,a.root/'e4_token_cache',a.rdd_bandwidth,a.draws,a.seed)
  rdd_path=a.root/f'e4_rdd_placebo_rows_{a.tag}.parquet';rdd_summary_path=a.root/f'e4_rdd_placebo_summary_{a.tag}.csv'
  if rdd_path.exists() or rdd_summary_path.exists(): raise FileExistsError(rdd_path)
  rdd_detail.to_parquet(rdd_path,index=False);rdd_summary.to_csv(rdd_summary_path,index=False)
 print(json.dumps({'rows':len(per),'primary_candidates':len(df),'excluded_missing_boundary':len(all_items)-len(df),'summary':str(out),'worst_case':str(worst) if a.worst_case else None},sort_keys=True))
if __name__=='__main__':main()
