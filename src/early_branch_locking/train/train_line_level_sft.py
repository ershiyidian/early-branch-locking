#!/usr/bin/env python3
"""LoRA training for solver-derived GRPO-line Countdown SFT supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from early_branch_locking._repo import RLVR_DATA_ROOT, REPO_ROOT  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("verify-base", "train"), default="verify-base")
    p.add_argument("--tag", required=True)
    p.add_argument("--k", type=int, choices=(1, 2, 4, 8), required=True)
    p.add_argument("--sampling", choices=("entrance-diverse", "uniform"), default="entrance-diverse")
    p.add_argument("--base-model", type=Path, default=REPO_ROOT / "model" / "qwen253B")
    p.add_argument("--reference-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "TinyZero" / "external-countdown" / "philschmid_qwen-2.5-3b-r1-countdown" / "checkpoint-25")
    p.add_argument("--data-dir", type=Path, default=RLVR_DATA_ROOT / "outputs" / "grpo_sft")
    p.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints" / "grpo_line_sft")
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--warmup-ratio", type=float, default=.03)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--max-examples", type=int, default=0, help="Diagnostic-only cap; a capped run is never a formal SFT artifact.")
    p.add_argument("--supervision-path", type=Path, default=None,
                   help="Explicit MLE JSONL source; required for non-default M12 factor data.")
    p.add_argument("--prompt-style", choices=("legacy", "native"), default="legacy",
                   help="Prompt contract. M12 uses native public Countdown messages.")
    return p.parse_args(argv)


def data_path(a):
    return a.supervision_path or a.data_dir / f"grpo_line_sft_supervision_k{a.k}_{a.sampling}_v1.jsonl"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify(a):
    from transformers import AutoConfig, AutoTokenizer
    base_cfg, ref_cfg = AutoConfig.from_pretrained(str(a.base_model), trust_remote_code=True), AutoConfig.from_pretrained(str(a.reference_checkpoint), trust_remote_code=True)
    base_tok, ref_tok = AutoTokenizer.from_pretrained(str(a.base_model), trust_remote_code=True, use_fast=False), AutoTokenizer.from_pretrained(str(a.reference_checkpoint), trust_remote_code=True, use_fast=False)
    result = {"base_model": str(a.base_model), "reference_checkpoint": str(a.reference_checkpoint),
              "model_type_match": base_cfg.model_type == ref_cfg.model_type, "vocab_size_match": base_cfg.vocab_size == ref_cfg.vocab_size,
              "tokenizer_vocab_match": base_tok.get_vocab() == ref_tok.get_vocab(), "base_vocab_size": base_cfg.vocab_size,
              "reference_vocab_size": ref_cfg.vocab_size,
              "known_limitation": "config/tokenizer equivalence does not prove pre-GRPO weight equivalence"}
    if not all(result[k] for k in ("model_type_match", "vocab_size_match", "tokenizer_vocab_match")):
        raise RuntimeError(result)
    print(json.dumps(result, sort_keys=True))


def train(a):
    # The scheduler may inherit a stale visibility mask.  This argument names
    # the requested physical device, so bind it before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu_id)
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from early_branch_locking.countdown.public_grpo_replication import build_philschmid_messages
    path = data_path(a)
    if not path.exists():
        raise FileNotFoundError(path)
    output = a.checkpoint_root / a.tag
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    source_rows = len(rows)
    if a.max_examples:
        if a.max_examples < 1:
            raise ValueError("--max-examples must be positive")
        rows = rows[:a.max_examples]
    if not rows:
        raise ValueError("empty supervision")
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(a.base_model), trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    def encode_example(r):
        messages = build_philschmid_messages(r["numbers"], int(r["target"])) if a.prompt_style == "native" else [
            {"role":"system", "content":"You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
            {"role":"user", "content":f"Using the numbers {r['numbers']}, create an equation that equals {r['target']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags."},
            {"role":"assistant", "content":"Let me solve this step by step.\n<think>"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        prompt_ids=tokenizer.encode(prompt,add_special_tokens=False); target_ids=tokenizer.encode(r['completion'],add_special_tokens=False)
        if tokenizer.eos_token_id is not None: target_ids.append(tokenizer.eos_token_id)
        return prompt_ids,target_ids
    # Supervision must remain whole: partial answers change both the native
    # format target and the intended leaf distribution. Fail before model load.
    lengths=[]
    for ix,r in enumerate(rows):
        prompt_ids,target_ids=encode_example(r); total_length=len(prompt_ids)+len(target_ids);lengths.append(total_length)
        if len(prompt_ids) >= a.max_seq_len or total_length > a.max_seq_len:
            raise ValueError(f"sequence exceeds max_seq_len at example {ix}: prompt={len(prompt_ids)} target={len(target_ids)} total={total_length} limit={a.max_seq_len}")
    class Set(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, ix):
            r = rows[ix]
            prompt_ids,target_ids=encode_example(r)
            return prompt_ids+target_ids,[-100]*len(prompt_ids)+target_ids
    def collate(batch):
        width=max(len(x[0]) for x in batch); pad=tokenizer.pad_token_id
        return {"input_ids":torch.tensor([x[0]+[pad]*(width-len(x[0])) for x in batch]), "labels":torch.tensor([x[1]+[-100]*(width-len(x[1])) for x in batch]), "attention_mask":torch.tensor([[1]*len(x[0])+[0]*(width-len(x[0])) for x in batch])}
    model=AutoModelForCausalLM.from_pretrained(str(a.base_model),torch_dtype=torch.bfloat16,trust_remote_code=True,low_cpu_mem_usage=True).cuda()
    model.gradient_checkpointing_enable(); model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=a.lora_rank,lora_alpha=a.lora_alpha,lora_dropout=0.,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    loader=DataLoader(Set(),batch_size=a.batch_size,shuffle=True,collate_fn=collate)
    total_batches=max(1,math.ceil(len(loader)*a.epochs)); total=max(1,math.ceil(total_batches/a.grad_accum))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr); sched=get_cosine_schedule_with_warmup(opt,int(total*a.warmup_ratio),total)
    output.mkdir(parents=True); step=batches_seen=0; losses=[]; opt.zero_grad(set_to_none=True)
    for _epoch in range(math.ceil(a.epochs)):
        for batch in loader:
            if batches_seen >= total_batches: break
            batch={k:v.cuda() for k,v in batch.items()}; loss=model(**batch).loss/a.grad_accum; loss.backward(); losses.append(float(loss.item())*a.grad_accum); batches_seen+=1
            if batches_seen % a.grad_accum and batches_seen != total_batches: continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step();sched.step();opt.zero_grad(set_to_none=True);step+=1
            if step % a.save_every == 0 or step == total:
                ckpt=output/f"step_{step:05d}"; ckpt.mkdir();model.save_pretrained(str(ckpt));tokenizer.save_pretrained(str(ckpt))
        if batches_seen >= total_batches: break
    manifest={"artifact":"GRPO-line SFT LoRA","status":"diagnostic" if a.max_examples else "complete","formal_complete":not bool(a.max_examples),"tag":a.tag,"data":str(path),"data_sha256":sha(path),"source_rows":source_rows,"selected_rows":len(rows),"base_model":str(a.base_model),"reference_checkpoint":str(a.reference_checkpoint),"config":{"k":a.k,"sampling":a.sampling,"prompt_style":a.prompt_style,"epochs":a.epochs,"lr":a.lr,"effective_batch":a.batch_size*a.grad_accum,"lora_rank":a.lora_rank,"lora_alpha":a.lora_alpha,"bf16":True,"gradient_checkpointing":True,"save_every":a.save_every,"seed":a.seed,"max_examples":a.max_examples},"sequence_length":{"max":max(lengths),"min":min(lengths),"max_seq_len":a.max_seq_len,"truncation_policy":"forbidden"},"batches_seen":batches_seen,"target_batches":total_batches,"optimizer_steps":step,"target_optimizer_steps":total,"final_loss":losses[-1] if losses else None,"checkpoint_root":str(output)}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({"checkpoint_root":str(output),"steps":step},sort_keys=True))


def main(argv=None):
    a=parse_args(argv); verify(a) if a.mode == "verify-base" else train(a)
if __name__ == "__main__": main()
