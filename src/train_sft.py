"""RFT 데이터로 LoRA SFT. 학습용 venv에서 실행한다.

  /workspace/venv-train/bin/python src/train_sft.py \
      --data out/sft_r1.jsonl --model ./qwen3b --out ckpt/sft_r1 --merge

설계 원칙
  1. 학습 형식 = 추론 형식. chat template 을 그대로 씌우고, loss 는 assistant
     구간에만 건다. 이 둘이 어긋나면 손실은 잘 떨어지는데 성능만 나빠진다.
  2. LoRA. full FT 대비 다양성 보존이 낫다는 보고가 있고, 무엇보다 되돌리기 쉽다.
  3. 1 epoch 기본. 자기 출력으로 하는 학습에서 여러 epoch 는 붕괴를 부른다.
  4. 쉬운 문항 하향표집 옵션. RFT 데이터의 74%가 모델이 이미 절반 이상 맞히는
     문항에서 나오므로, 그대로 쓰면 아는 것을 반복 학습하며 다양성을 잃는다.
"""
import argparse, json, math, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from prompts import build_messages, DEFAULT


def load_rows(path, pass_at_k=None, easy_keep=1.0, max_rows=0, seed=42):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n0 = len(rows)
    dropped = 0
    if pass_at_k and Path(pass_at_k).exists() and easy_keep < 1.0:
        pk = json.loads(Path(pass_at_k).read_text())
        k = max(pk.values()) if pk else 16
        rng = random.Random(seed)
        keep = []
        for r in rows:
            n = pk.get(r["id"])
            easy = n is not None and k and (n / k) >= 0.5
            if easy and rng.random() > easy_keep:
                dropped += 1
                continue
            keep.append(r)
        rows = keep
    if max_rows:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_rows]
    print(f"[data] {n0:,} -> {len(rows):,}"
          + (f" (쉬운 문항 {dropped:,}건 제외)" if dropped else ""))
    return rows


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", default="out/sft_r1.jsonl")
    ap.add_argument("--model", default="./qwen3b")
    ap.add_argument("--out", default="ckpt/sft_r1")
    ap.add_argument("--pass-at-k", default="out/pass_at_k_r1.json")
    ap.add_argument("--easy-keep", type=float, default=1.0,
                    help="쉬운 문항(정답률>=50%) 중 남길 비율. 1.0=전부")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true",
                    help="학습 후 병합본 저장 (vLLM 이 바로 읽을 수 있는 형태)")
    a = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model
    set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_rows(a.data, a.pass_at_k, a.easy_keep, a.max_rows, a.seed)

    # ── 토큰화: 프롬프트 구간은 loss 에서 제외 ──────────────
    feats, skipped = [], 0
    for r in rows:
        msgs = build_messages(r["question"], DEFAULT)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + r["solution"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full,   add_special_tokens=False)["input_ids"]
        if len(f_ids) > a.max_len:
            skipped += 1
            continue
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        feats.append({"input_ids": f_ids, "labels": labels})
    print(f"[data] 토큰화 완료 {len(feats):,}건 (길이 초과 {skipped}건 제외)")
    if not feats:
        sys.exit("학습 표본이 없다.")
    lens = sorted(len(f["input_ids"]) for f in feats)
    print(f"[data] 길이 중앙값 {lens[len(lens)//2]} · 95분위 {lens[int(len(lens)*.95)]} · 최대 {lens[-1]}")

    def collate(batch):
        mx = max(len(b["input_ids"]) for b in batch)
        ids, lbl, att = [], [], []
        for b in batch:
            pad = mx - len(b["input_ids"])
            ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            lbl.append(b["labels"] + [-100] * pad)
            att.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lbl),
                "attention_mask": torch.tensor(att)}

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    steps = math.ceil(len(feats) / (a.bs * a.accum) * a.epochs)
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=a.warmup,
        logging_steps=max(1, steps // 40), save_strategy="no",
        bf16=True, gradient_checkpointing=True, report_to=[],
        seed=a.seed, remove_unused_columns=False, max_grad_norm=1.0)
    print(f"[train] {len(feats):,}건 · 유효배치 {a.bs*a.accum} · {steps} step · lr {a.lr}")

    Trainer(model=model, args=args, train_dataset=feats,
            data_collator=collate).train()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"[save] 어댑터 {a.out}")

    meta = {"data": a.data, "n_train": len(feats), "epochs": a.epochs, "lr": a.lr,
            "rank": a.rank, "alpha": a.alpha, "easy_keep": a.easy_keep,
            "max_len": a.max_len, "eff_batch": a.bs * a.accum, "steps": steps}
    (Path(a.out) / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    if a.merge:
        md = a.out + "-merged"
        print(f"[merge] {md} …")
        merged = model.merge_and_unload()
        merged = merged.to(torch.bfloat16)
        merged.save_pretrained(md, safe_serialization=True)
        tok.save_pretrained(md)
        print(f"[merge] 완료. 검증:\n"
              f"  python3 src/baseline.py --model {md} --val out/val.csv --limit 300 \\\n"
              f"      --prompts C --modes chat --k 64 --temp 0.8 --max-tokens 3072")


if __name__ == "__main__":
    main()
