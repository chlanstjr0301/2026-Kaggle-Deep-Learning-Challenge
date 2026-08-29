#!/usr/bin/env python3
"""DPO —— 오답을 대조 신호로 쓰는 유일한 실험. 학습 표의 마지막 빈칸."""
import argparse, json, re, sys, math
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--pairs", default="out/dpo_pairs.jsonl")
ap.add_argument("--model", default="./qwen3b")
ap.add_argument("--out",   default="ckpt/dpo_r1")
ap.add_argument("--lr",    type=float, default=5e-6)
ap.add_argument("--beta",  type=float, default=0.1)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--bs",    type=int, default=2)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--max-prompt", type=int, default=640)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--seed",  type=int, default=42)
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()

# 생성 때 쓴 프롬프트를 같은 모듈에서 직접 가져온다 (prompts.py 는 main() 이 없어 임포트 안전)
sys.path.insert(0, "src")
from prompts import PROMPTS, SYSTEM, build_messages
TMPL = PROMPTS["C"]
print(f"[prompt] C = {TMPL[:90]!r}...")

rows = [json.loads(l) for l in open(a.pairs)]
def fmt(q):
    try: return TMPL.format(q=q)
    except (KeyError, IndexError): return TMPL.replace("{q}", str(q))
ds = Dataset.from_list([{
    "prompt":   [{"role": "system",    "content": SYSTEM},
                 {"role": "user",      "content": fmt(r["prompt"])}],
    "chosen":   [{"role": "assistant", "content": r["chosen"]}],
    "rejected": [{"role": "assistant", "content": r["rejected"]}],
} for r in rows])
print(f"[data] {len(ds):,}쌍")

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False

steps = math.ceil(len(ds) / (a.bs * a.accum) * a.epochs)
cfg = DPOConfig(
    output_dir=a.out, num_train_epochs=a.epochs,
    per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
    learning_rate=a.lr, beta=a.beta, lr_scheduler_type="cosine", warmup_steps=max(1, steps // 20),
    max_length=a.max_len,
    logging_steps=max(1, steps // 40), save_strategy="no",
    bf16=True, gradient_checkpointing=True, report_to=[], seed=a.seed, max_grad_norm=1.0)
lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])
print(f"[train] {len(ds):,}쌍 · 유효배치 {a.bs*a.accum} · {steps} step · lr {a.lr} · beta {a.beta}")
tr = DPOTrainer(model=model, ref_model=None, args=cfg,
                train_dataset=ds, processing_class=tok, peft_config=lora)
tr.train()
Path(a.out).mkdir(parents=True, exist_ok=True)
tr.model.save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump({"pairs": len(ds), "lr": a.lr, "beta": a.beta, "steps": steps,
           "rank": a.rank, "epochs": a.epochs},
          open(Path(a.out)/"run_meta.json","w"), ensure_ascii=False, indent=2)
print(f"[save] {a.out}")
if a.merge:
    md = a.out + "-merged"
    print(f"[merge] {md} …")
    mm = tr.model.merge_and_unload().to(torch.bfloat16)
    mm.save_pretrained(md, safe_serialization=True); tok.save_pretrained(md)
    print("[merge] 완료")
