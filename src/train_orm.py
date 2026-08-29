"""검증기(ORM) LoRA 학습. 학습용 venv 에서 실행한다.

  /workspace/venv-train/bin/python src/train_orm.py \
      --data out/orm_r1.jsonl --model ./qwen3b --out ckpt/orm_r1 --merge

형식: 채점 프롬프트를 chat template 으로 씌우고, **첫 assistant 토큰 하나**
(Yes / No)에만 loss 를 건다. 분류 헤드 없이 LoRA 만으로 끝나므로 vLLM 이
그대로 채점할 수 있고, 규정상으로도 "같은 베이스 + 우리 어댑터"로 깔끔하다.
"""
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from orm_prompt import build, YES, NO, DEFAULT_STANCE


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", default="out/orm_r1.jsonl")
    ap.add_argument("--model", default="./qwen3b")
    ap.add_argument("--out", default="ckpt/orm_r1")
    ap.add_argument("--stance", default=DEFAULT_STANCE)
    ap.add_argument("--maj-wrong-weight", type=float, default=1.0,
                    help="다수결이 틀린 문항의 표본 가중. 1.0=동일")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model
    set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    yes_ids = tok(YES, add_special_tokens=False)["input_ids"]
    no_ids = tok(NO, add_special_tokens=False)["input_ids"]
    if len(yes_ids) != 1 or len(no_ids) != 1:
        sys.exit(f"Yes/No 가 단일 토큰이 아니다: {yes_ids} {no_ids}. orm_prompt 를 조정할 것.")
    YID, NID = yes_ids[0], no_ids[0]
    print(f"[tok] Yes={YID} No={NID}")

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    feats, skipped = [], 0
    for r in rows:
        prompt = build(tok, r["question"], r["solution"], a.stance)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(p_ids) + 1 > a.max_len:
            skipped += 1
            continue
        ids = p_ids + [YID if r["label"] else NID]
        labels = [-100] * len(p_ids) + [ids[-1]]      # 마지막 한 토큰에만 loss
        w = a.maj_wrong_weight if r.get("maj_wrong") else 1.0
        feats.append({"input_ids": ids, "labels": labels, "w": w})
    print(f"[data] {len(rows):,} -> {len(feats):,} (길이 초과 {skipped}건 제외)")
    if not feats:
        sys.exit("학습 표본 없음")
    lens = sorted(len(f["input_ids"]) for f in feats)
    print(f"[data] 길이 중앙값 {lens[len(lens)//2]} · 95분위 {lens[int(len(lens)*.95)]}")

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
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    steps = math.ceil(len(feats) / (a.bs * a.accum) * a.epochs)
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=max(1, steps // 40), save_strategy="no",
        bf16=True, gradient_checkpointing=True, report_to=[],
        seed=a.seed, remove_unused_columns=False, max_grad_norm=1.0)
    print(f"[train] {len(feats):,}건 · 유효배치 {a.bs*a.accum} · {steps} step · lr {a.lr}")

    Trainer(model=model, args=args, train_dataset=feats, data_collator=collate).train()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    (Path(a.out) / "run_meta.json").write_text(json.dumps(
        {"data": a.data, "n": len(feats), "stance": a.stance, "lr": a.lr,
         "rank": a.rank, "epochs": a.epochs, "yes_id": YID, "no_id": NID},
        ensure_ascii=False, indent=2))
    print(f"[save] {a.out}")

    if a.merge:
        md = a.out + "-merged"
        merged = model.merge_and_unload().to(torch.bfloat16)
        merged.save_pretrained(md, safe_serialization=True); tok.save_pretrained(md)
        print(f"[merge] {md}")


if __name__ == "__main__":
    main()
