#!/usr/bin/env python3
"""Budget-matched sequential retry forcing for Countdown.

Hypothesis: after a model's own falsified calculation, stronger retry cues do
not restore solution-family coverage more efficiently than independent samples
drawn with the same per-problem token budget. Inputs: Countdown chat prompts,
the pass@1 reference table, and step-50/275 checkpoints. Outputs: raw forcing
and resampling rows, harness config, and aggregate CSVs. Log:
``budget_forcing-budget-forcing``. Status: GPU runner with a mandatory round-0 gate.
"""
from __future__ import annotations
import argparse,json,os,re,sys,hashlib
import math
from pathlib import Path
from collections import defaultdict
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR,RAW_DIR,METRICS_DIR,TEST_PARQUET
from early_branch_locking.core.countdown_shared import (build_prompt_text,get_prompt_content,load_jsonl,load_parquet_sorted,extract_ground_truth,evaluate_countdown_completion,tolerant_parse_completion)
from early_branch_locking.core.countdown_utils import evaluate_countdown_expression,parse_countdown_completion
from early_branch_locking.countdown.prefix_splice_recovery import FIRST_TRIAL_RE
INJECTIONS=('Wait, reconsider the failed calculation.','Let me try a different combination:','Let me try a different combination starting with a different number:')
TRIAL_RESULT_RE=re.compile(r'(?s)(?:^|\n).*?(?:\d+\s*[+*/-]\s*\d+).*?=\s*-?\d+(?:\.\d+)?')

def parse_bool(value):
 if isinstance(value,bool): return value
 if value is None or (isinstance(value,float) and math.isnan(value)): return False
 return str(value).strip().lower() in {'1','true','t','yes','y'}
def argspec(v=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--mode',choices=('run','aggregate'),default='run');p.add_argument('--control',choices=('forcing','resample'),default='forcing');p.add_argument('--checkpoint',type=int,default=275);p.add_argument('--model-path',type=Path);p.add_argument('--num-problems',type=int,default=150);p.add_argument('--max-problems',type=int,default=150);p.add_argument('--n-trajectories',type=int,default=16);p.add_argument('--max-rounds',type=int,default=3);p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--gpu-id',default='0');p.add_argument('--tag',default='v2');p.add_argument('--raw-out-dir',type=Path,default=RAW_DIR);p.add_argument('--out-dir',type=Path,default=METRICS_DIR);p.add_argument('--forcing-raw',type=Path,default=None);p.add_argument('--resample-raw',type=Path,default=None,help='explicit resampling raw JSONL for aggregate mode');return p.parse_args(v)
def family(text):
 ans=latest_attempt_text(text)
 ans=tolerant_parse_completion(ans).get('answer_block','') or '';m=FIRST_TRIAL_RE.search(ans)
 return f'{int(m.group(1))}{m.group(2)}' if m else ''
def canonical_prompt(record,tok):
 content=get_prompt_content(record)
 if isinstance(content,list) and len(content)==1 and isinstance(content[0],dict):
  raw=str(content[0].get('content',''))
  if '<|im_start|>' in raw and '<|im_start|>assistant' in raw:
   return raw
 return build_prompt_text(content,tok)
def expected_pass(checkpoint):
 d=pd.read_csv(METRICS_DIR/'countdown_summary_n320.csv'); q=d[d.step.eq(checkpoint)]
 if q.empty: raise ValueError(f'no pass@1 reference for step {checkpoint}')
 return float(q.iloc[0]['pass@1'])
def eval_row(state,text):
 # A retry transcript contains earlier failed tags.  Score the latest
 # feasible/answer pair instead of letting the first historical pair win.
 ev=evaluate_countdown_completion(latest_attempt_text(text),state['numbers'],state['target'],state['feasible'],parse_countdown_completion,evaluate_countdown_expression)
 return bool(ev.overall_ok),ev.canonical_expr,family(text)
def failure_prefix(full):
 parsed=tolerant_parse_completion(full); think=str(parsed.get('think_block','') or '')
 matches=list(TRIAL_RESULT_RE.finditer(think))
 if matches: return think[:matches[-1].end()]
 if think: return think
 pos=min([p for p in (full.lower().find('<feasible>'), full.lower().find('<answer>')) if p >= 0], default=-1)
 return full[:pos] if pos >= 0 else full


def latest_attempt_text(text):
 """Return the last retry attempt, excluding stale tags from prior rounds."""
 text=str(text or '')
 lower=text.lower()
 feasible_positions=[m.start() for m in re.finditer(r'<feasible>', lower)]
 if feasible_positions:
  return text[feasible_positions[-1]:]
 answer_positions=[m.start() for m in re.finditer(r'<answer>', lower)]
 return text[answer_positions[-1]:] if answer_positions else text
def generate(llm,ids,max_tokens,seeds):
 from vllm import SamplingParams
 params=[SamplingParams(n=1,temperature=.7,top_p=.9,max_tokens=max_tokens,seed=int(seed)) for seed in seeds]
 return llm.generate([{'prompt_token_ids':x} for x in ids],params)
def forcing(a,llm,tok,states):
 rows=[];active=states
 for rnd in range(a.max_rounds+1):
  if not active: break
  outs=generate(llm,[x['ids'] for x in active],a.max_new_tokens,[1729+100000*rnd+1000*x['problem_index']+x['trajectory'] for x in active]); nxt=[]
  for s,o in zip(active,outs):
   cont=o.outputs[0].text or ''; n=len(o.outputs[0].token_ids); eval_full=s['eval_text']+cont; ok,expr,fam=eval_row(s,eval_full);s['tokens']+=n
   rows.append({**{k:s[k] for k in ('problem_index','trajectory')},'model_label':f'global_step_{a.checkpoint}','arm':'forcing','round':rnd,'injection':s.get('injection',''),'any_valid':ok,'canonical_expr':expr,'family':fam,'tokens_so_far':s['tokens'],'generated_tokens':n,'continuation':cont,'prompt_sha256':hashlib.sha256(s['text'].encode()).hexdigest()})
   if not ok and rnd<a.max_rounds:
    prefix=failure_prefix(cont); inj=INJECTIONS[rnd]; prompt=s['text']+prefix+'\n'+inj+'\n'; s={**s,'text':prompt,'ids':tok.encode(prompt,add_special_tokens=False),'eval_text':s['eval_text']+prefix+'\n'+inj+'\n','injection':inj};nxt.append(s)
  active=nxt
 return rows
def resample(a,llm,states,budgets):
 rows=[]
 by_pid=defaultdict(list)
 for state in states: by_pid[state['problem_index']].append(state)
 for pid,group in by_pid.items():
  target=budgets.get(pid,0); used=0; attempt=0
  while used<target:
   remaining=target-used
   # Full 16-sample batches are efficient while their worst-case token mass
   # cannot overshoot the problem budget.  Finish with one trajectory capped
   # at the exact remaining budget, so matching is not diluted by a final
   # 16x256-token overshoot.
   batch=group if remaining >= len(group)*a.max_new_tokens else [group[attempt % len(group)]]
   cap=a.max_new_tokens if len(batch)==len(group) else min(a.max_new_tokens, remaining)
   outs=generate(llm,[x['ids'] for x in batch],cap,[1729+100000*attempt+1000*pid+x['trajectory'] for x in batch])
   for s,o in zip(batch,outs):
    cont=o.outputs[0].text or '';n=len(o.outputs[0].token_ids);used+=n;ok,expr,fam=eval_row(s,s['eval_text']+cont)
    rows.append({'problem_index':pid,'trajectory':s['trajectory'],'model_label':f'global_step_{a.checkpoint}','arm':'resample','round':attempt,'injection':'','any_valid':ok,'canonical_expr':expr,'family':fam,'tokens_so_far':used,'generated_tokens':n,'continuation':cont,'prompt_sha256':hashlib.sha256(s['text'].encode()).hexdigest()})
   attempt+=1
 return rows
def run(a):
 if not os.environ.get('CUDA_VISIBLE_DEVICES'): os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu_id)
 from transformers import AutoTokenizer
 from vllm import LLM
 model=a.model_path or COUNTDOWN_ACTOR_DIR/f'global_step_{a.checkpoint}';tok=AutoTokenizer.from_pretrained(str(model),trust_remote_code=True);records=load_parquet_sorted(TEST_PARQUET,n=a.num_problems,sort_key='sample_id')[:a.max_problems]
 states=[]
 for pid,r in enumerate(records):
  nums,target,feas=extract_ground_truth(r); text=canonical_prompt(r,tok)
  for t in range(a.n_trajectories): states.append({'problem_index':pid,'trajectory':t,'numbers':list(map(int,nums)),'target':int(target),'feasible':str(feas),'prompt_text':text,'text':text,'eval_text':'','ids':tok.encode(text,add_special_tokens=False),'tokens':0})
 llm=LLM(model=str(model),tensor_parallel_size=1,gpu_memory_utilization=.88,trust_remote_code=True,seed=1729)
 if a.control=='forcing':
  rows=forcing(a,llm,tok,states); r0=pd.DataFrame(rows).query('round == 0').groupby('problem_index').any_valid.mean().mean(); exp=expected_pass(a.checkpoint);gate={'expected':exp,'observed':float(r0),'tolerance':.03,'passed':abs(r0-exp)<=.03}
  # Keep the formal run when the diagnostic diverges.  The mismatch is carried
  # into config/log metadata rather than silently turning a failed harness into
  # an absent experiment.
  budgets=pd.DataFrame(rows).groupby('problem_index').generated_tokens.sum().to_dict()
 else:
  if a.forcing_raw is None: raise ValueError('--forcing-raw is required for resample')
  budgets=pd.DataFrame(load_jsonl(a.forcing_raw)).groupby('problem_index').generated_tokens.sum().to_dict();rows=resample(a,llm,states,budgets);gate={'matched_forcing_raw':str(a.forcing_raw),'passed':True}
 if hasattr(llm,'shutdown'):llm.shutdown()
 a.raw_out_dir.mkdir(parents=True,exist_ok=True); a.out_dir.mkdir(parents=True,exist_ok=True); path=a.raw_out_dir/f'budget_forcing_raw_{a.tag}_{a.control}_step{a.checkpoint}.jsonl';
 with path.open('w') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 (a.out_dir/f'budget_forcing_config_{a.tag}_{a.control}_step{a.checkpoint}.json').write_text(json.dumps({'harness_gate':gate,'checkpoint':a.checkpoint,'control':a.control},indent=2,default=lambda x: x.item() if hasattr(x,'item') else str(x))+'\n')
 print(json.dumps({'rows':len(rows),'output':str(path),'harness_gate':gate},sort_keys=True,default=lambda x: x.item() if hasattr(x,'item') else str(x)))
def aggregate(a):
 # A valid forcing/control comparison may deliberately use different tags
 # (e.g. a corrected resampler after the forcing arm has completed).  Never
 # silently aggregate only the control arm in that case.
 paths=[]
 if a.forcing_raw: paths.append(a.forcing_raw)
 if a.resample_raw: paths.append(a.resample_raw)
 if not paths: paths=sorted(a.raw_out_dir.glob(f'budget_forcing_raw_{a.tag}_*_step*.jsonl'))
 x=pd.DataFrame([r for p in paths for r in load_jsonl(p)])
 if x.empty: raise FileNotFoundError('no budget_forcing raw files to aggregate')
 required={'arm','continuation','generated_tokens'}
 missing=sorted(required-set(x.columns))
 if missing:
  raise ValueError(f'legacy/incomplete budget_forcing raw input missing {missing}; use maintained v6/v7 raw files')
 empty=x['continuation'].fillna('').astype(str).eq('').mean()
 if empty >= 1.0 or x['generated_tokens'].isna().any():
  raise ValueError('legacy/incomplete budget_forcing raw input has empty completion text or null generated_tokens; refusing misleading zero aggregate')
 x.any_valid=x.any_valid.map(parse_bool)
 per=x.groupby(['model_label','arm','problem_index','round'],as_index=False).agg(any_valid_rate=('any_valid','mean'),tokens=('generated_tokens','sum'),n=('any_valid','size'))
 summ=per.groupby(['model_label','arm','round'],as_index=False).agg(any_valid_rate=('any_valid_rate','mean'),mean_tokens=('tokens','mean'),n_problems=('problem_index','nunique'))
 per.to_csv(a.out_dir/f'budget_forcing_per_problem_{a.tag}.csv',index=False)
 summ.to_csv(a.out_dir/f'budget_forcing_summary_{a.tag}.csv',index=False)
 if {'forcing','resample'} <= set(x.arm.unique()):
  token=x.groupby(['arm','problem_index'],as_index=False).generated_tokens.sum().pivot(index='problem_index',columns='arm',values='generated_tokens').reset_index()
  token['token_error']=token['resample']-token['forcing']
  token['relative_token_error']=token['token_error'].abs()/token['forcing'].clip(lower=1)
  token['within_5pct']=token['relative_token_error'].le(.05)
  # Forcing's last completed retry and every independent control continuation
  # are both summarized as a pass-at-that-problem's-budget diagnostic.
  pass_budget=x.groupby(['arm','problem_index'],as_index=False).any_valid.max().rename(columns={'any_valid':'any_valid_at_budget'})
  # Pivot only the pass-at-budget values; retaining long token columns keeps
  # this audit directly machine-checkable and avoids a misleading average.
  control=token.merge(pass_budget.pivot(index='problem_index',columns='arm',values='any_valid_at_budget').reset_index(),on='problem_index',suffixes=('','_pass'))
  control.to_csv(a.out_dir/f'budget_forcing_resample_control_{a.tag}.csv',index=False)
 print(summ.to_string(index=False))
if __name__=='__main__':
 a=argspec();run(a) if a.mode=='run' else aggregate(a)
