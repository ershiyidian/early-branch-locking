#!/usr/bin/env python3
"""Full blinded semantic-boundary annotation over the same-trace score pool."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from early_branch_locking.core.api_judge import call_boundary_judge, load_api_config, resolve_boundary_span  # noqa
from early_branch_locking.math_transfer.extract_solution_boundaries import _item_id, _units, origin_of  # noqa

DEFAULT_SCORES=ROOT/'data/rlvr/outputs/e2/qwen_7b_base_rl/scores.jsonl'
DEFAULT_OUT=ROOT/'data/rlvr/outputs/experiments/e4_semantic_boundary_v1'

def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('packet','judge','origin-rates','duplicate-agreement','export-spotcheck'),default='packet');p.add_argument('--scores',type=Path,default=DEFAULT_SCORES);p.add_argument('--out-dir',type=Path,default=DEFAULT_OUT);p.add_argument('--seed',type=int,default=1729);p.add_argument('--concurrency',type=int,default=8);p.add_argument('--max-attempts',type=int,default=6);p.add_argument('--timeout',type=float,default=60);p.add_argument('--max-items',type=int,default=0);p.add_argument('--api-model',default=None);p.add_argument('--bootstrap-draws',type=int,default=10000);return p.parse_args(argv)
def paths(a):return {'packet':a.out_dir/'packet_blind.jsonl','index':a.out_dir/'index_private.jsonl','annotations':a.out_dir/'semantic_boundary_annotations_v1.jsonl','rates':a.out_dir/'e4_origin_rates_v1.csv','duplicates':a.out_dir/'e4_duplicate_agreement_v1.csv','spotcheck':a.out_dir/'human_spotcheck_packet.jsonl','manifest':a.out_dir/'manifest.json'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def packet(a):
 ps=paths(a); rows=pd.read_json(a.scores,lines=True); rows=rows[rows.style_variant.astype(str).eq('raw')].copy()
 req={'question','completion','model','benchmark','problem_id'};missing=req-set(rows)
 if missing:raise ValueError(f'scores missing {sorted(missing)}')
 if ps['packet'].exists() or ps['index'].exists():raise FileExistsError('packet already exists')
 rng=random.Random(a.seed); ids=[]
 for row in rows.to_dict('records'):
  sid=int(row.get('sample_index',row.get('sample_id',0))); iid=_item_id(row,sid); ids.append((iid,row,sid))
 if len({x[0] for x in ids})!=len(ids):raise ValueError('nonunique item ids')
 duplicated={iid for iid,_,_ in rng.sample(ids,max(1,round(.10*len(ids))))}
 ps['packet'].parent.mkdir(parents=True,exist_ok=True);tp=ps['packet'].with_suffix('.partial');ti=ps['index'].with_suffix('.partial')
 with tp.open('w') as po,ti.open('w') as io:
  for iid,row,sid in ids:
   units=_units(str(row['completion'])); public={'item_id':iid,'question':str(row['question']),'response':str(row['completion']),'units':units}
   private={'item_id':iid,'benchmark':str(row['benchmark']),'origin':origin_of(str(row['model'])),'problem_id':str(row['problem_id']),'sample_index':sid,'parser_char_end':row.get('p1_char_end'),'duplicate':iid in duplicated}
   for duplicate_index in ((0,1) if iid in duplicated else (0,)):
    po.write(json.dumps({**public,'dup_index':duplicate_index},ensure_ascii=False,sort_keys=True)+'\n');io.write(json.dumps({**private,'dup_index':duplicate_index},ensure_ascii=False,sort_keys=True)+'\n')
 tp.replace(ps['packet']);ti.replace(ps['index']);payload={'artifact':'semantic_boundary_analysis full semantic packet','status':'complete','n_unique_items':len(ids),'n_annotation_requests':len(ids)+len(duplicated),'duplicate_fraction':.10,'scores_sha256':sha(a.scores),'packet_sha256':sha(ps['packet']),'blind_fields':['model','origin','parser_char_end','source_path'],'seed':a.seed}
 ps['manifest'].write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,sort_keys=True))
def judge(a):
 ps=paths(a); done=set()
 if ps['annotations'].exists():
  for line in ps['annotations'].open():
   if line.strip():
    r=json.loads(line);done.add((r['item_id'],int(r['dup_index'])))
 items=[json.loads(x) for x in ps['packet'].open() if x.strip()];todo=[x for x in items if (x['item_id'],int(x['dup_index'])) not in done]
 if a.max_items:todo=todo[:a.max_items]
 config=load_api_config(repo_root=ROOT,model_override=a.api_model);ps['annotations'].parent.mkdir(parents=True,exist_ok=True)
 def one(item):
  try:r=call_boundary_judge(item['question'],item['units'],config,max_attempts=a.max_attempts,timeout=a.timeout);annotation=r.get('annotation');status=r.get('status','ok');err=None
  except Exception as e:annotation=None;status='api_failed';err=type(e).__name__
  fallback={'has_complete_numeric_calculation':False,'unit_id':None,'span_text':None,'boundary_kind':'none','confidence':'low','reason_short':'API unavailable'}
  resolution=resolve_boundary_span(item['response'],item['units'],annotation or fallback)
  return {'item_id':item['item_id'],'dup_index':int(item['dup_index']),'annotation':annotation,**resolution.as_dict(),'api_model':config.model,'api_backend_mode':status,'error_type':err}
 with ps['annotations'].open('a') as f,ThreadPoolExecutor(max_workers=a.concurrency) as pool:
  for future in as_completed([pool.submit(one,x) for x in todo]):f.write(json.dumps(future.result(),sort_keys=True,default=str)+'\n');f.flush()
 print(json.dumps({'submitted':len(todo),'existing':len(done),'output':str(ps['annotations'])},sort_keys=True))
def complete_annotation_frame(annotations,index):
 keys=['item_id','dup_index']
 if annotations.duplicated(keys).any() or index.duplicated(keys).any():
  raise RuntimeError('complete annotation packet requires unique annotation and index keys')
 annotation_keys={(str(row.item_id),int(row.dup_index)) for row in annotations[keys].itertuples(index=False)}
 index_keys={(str(row.item_id),int(row.dup_index)) for row in index[keys].itertuples(index=False)}
 if annotation_keys!=index_keys:
  raise RuntimeError(f'complete annotation packet key mismatch: annotations={len(annotation_keys)} index={len(index_keys)} shared={len(annotation_keys & index_keys)}')
 return annotations.merge(index,on=keys,validate='one_to_one')
def origin_rates(a):
 ps=paths(a); ann=pd.read_json(ps['annotations'],lines=True);idx=pd.read_json(ps['index'],lines=True)
 df=complete_annotation_frame(ann,idx)
 rows=[]
 for (bench,origin),g in df.groupby(['benchmark','origin']):
  rows.append({'benchmark':bench,'origin':origin,'n':len(g),'resolver_failure_rate':float(g.resolver_status.ne('resolved').mean()),'no_calculation_rate':float(g.resolver_status.eq('no_calculation').mean()),'resolved_rate':float(g.resolver_status.eq('resolved').mean())})
 out=pd.DataFrame(rows);wide=out.pivot(index='benchmark',columns='origin',values='resolver_failure_rate')
 for bench in out.benchmark.unique():
  base=float(wide.loc[bench,'base_origin']);rl=float(wide.loc[bench,'rl_origin']);delta=base-rl;mask=out.benchmark.eq(bench)
  out.loc[mask,'base_minus_rl_resolver_failure_rate']=delta
  out.loc[mask,'absolute_resolver_failure_difference_pp']=abs(delta)*100
  out.loc[mask,'worst_case_sensitivity_required']=abs(delta)>.05
  out.loc[mask,'worst_case_trigger_rule']='absolute base-origin minus rl-origin resolver failure rate > 5 percentage points'
 if ps['rates'].exists():raise FileExistsError(ps['rates'])
 temp=ps['rates'].with_suffix('.partial');out.to_csv(temp,index=False);temp.replace(ps['rates']);print(ps['rates'])
def duplicate_agreement(a):
 ps=paths(a);ann=pd.read_json(ps['annotations'],lines=True);idx=pd.read_json(ps['index'],lines=True);df=complete_annotation_frame(ann,idx);counts=df.groupby('item_id').dup_index.nunique();ids=counts[counts.eq(2)].index;df=df[df.item_id.isin(ids)].copy()
 if len(ids)==0:raise ValueError('no completed duplicate annotation pairs')
 piv=df.set_index(['item_id','dup_index']).unstack('dup_index')
 def pair(column):return piv[(column,0)],piv[(column,1)]
 left_status,right_status=pair('resolver_status');left_unit,right_unit=pair('resolved_unit_id');left_span,right_span=pair('resolved_span_text')
 both=left_status.eq('resolved')&right_status.eq('resolved');out=piv.index.to_frame(index=False).merge(df[df.dup_index.eq(0)][['item_id','benchmark','origin','problem_id']],on='item_id',validate='one_to_one')
 out['resolver_status_agreement']=left_status.eq(right_status).to_numpy(float);out['both_resolved']=both.to_numpy(float);out['resolved_unit_agreement']=np.where(both,left_unit.eq(right_unit),np.nan);out['resolved_span_agreement']=np.where(both,left_span.eq(right_span),np.nan)
 rows=[];rng=np.random.default_rng(a.seed)
 for keys,g in [('all',out),*list(out.groupby(['benchmark','origin'],sort=True))]:
  bench,origin=('all','all') if keys=='all' else keys
  for metric in ('resolver_status_agreement','both_resolved','resolved_unit_agreement','resolved_span_agreement'):
   problem=g.groupby('problem_id')[metric].mean().dropna().to_numpy(float)
   if len(problem):
    sampled=problem[rng.integers(len(problem),size=(a.bootstrap_draws,len(problem)))].mean(1);mean,lo,hi=float(problem.mean()),float(np.quantile(sampled,.025)),float(np.quantile(sampled,.975));reason=''
   else:mean=lo=hi=np.nan;reason='no_eligible_duplicate_problem_clusters'
   rows.append({'benchmark':bench,'origin':origin,'metric':metric,'mean':mean,'ci_lo':lo,'ci_hi':hi,'n_duplicate_pairs':len(g),'n_problems':len(problem),'not_estimable_reason':reason,'bootstrap_draws':a.bootstrap_draws,'bootstrap_seed':a.seed,'statistical_unit':'problem_cluster'})
 if ps['duplicates'].exists():raise FileExistsError(ps['duplicates'])
 temp=ps['duplicates'].with_suffix('.partial');pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(ps['duplicates']);print(ps['duplicates'])
def spotcheck(a):
 ps=paths(a);ann=pd.read_json(ps['annotations'],lines=True);idx=pd.read_json(ps['index'],lines=True);packet=pd.read_json(ps['packet'],lines=True);df=packet.merge(complete_annotation_frame(ann,idx),on=['item_id','dup_index'],validate='one_to_one');df[df.dup_index.eq(0)].sample(min(50,len(df)),random_state=a.seed)[['item_id','question','response','units','annotation','resolved_span_text','parser_char_end','benchmark','origin']].to_json(ps['spotcheck'],orient='records',lines=True,force_ascii=False);print(ps['spotcheck'])
def main(argv=None):
 a=parse(argv);{'packet':packet,'judge':judge,'origin-rates':origin_rates,'duplicate-agreement':duplicate_agreement,'export-spotcheck':spotcheck}[a.mode](a)
if __name__=='__main__':main()
