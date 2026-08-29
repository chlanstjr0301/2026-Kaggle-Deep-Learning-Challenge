"""검증기(ORM)로 후보 풀이를 채점한다. 생성 없이 prefill 한 번씩.

점수 = log P(Yes) − log P(No)  (첫 assistant 토큰에서)
분류 헤드가 없으므로 vLLM 이 그대로 돌린다.

  python3 src/score_orm.py --orm ckpt/orm_r1-merged \
      --gen out/gen_qwen3b__val_C_chat_k32_t0.8_mt3072.jsonl --limit 300
"""
import argparse, json, os, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from orm_prompt import build, YES, NO, DEFAULT_STANCE

FLOOR = -20.0          # 상위 logprobs 에 없을 때 쓰는 하한


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--orm", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stance", default=DEFAULT_STANCE)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=50, help="문항 단위 체크포인트")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or f"out/orm__{Path(a.orm).name}__{Path(a.gen).stem}.jsonl"
    keep = None
    if a.limit:
        df = pd.read_csv(a.val).head(a.limit)
        keep = set(df["id"])

    rows = []
    with open(a.gen, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if keep is None or r["id"] in keep:
                rows.append((r["id"], r["outputs"]))
    if not rows:
        sys.exit("채점할 항목 없음")

    done = set()
    if Path(out).exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [(i, o) for i, o in rows if i not in done]
    print(f"[orm] 전체 {len(rows)} · 완료 {len(done)} · 남음 {len(todo)}", flush=True)
    if not todo:
        print(f"이미 완료: {out}")
        return

    q = dict(zip(pd.read_csv(a.val)["id"], pd.read_csv(a.val)["question"]))

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.orm, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_len, seed=42, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    yid = tok(YES, add_special_tokens=False)["input_ids"]
    nid = tok(NO, add_special_tokens=False)["input_ids"]
    if len(yid) != 1 or len(nid) != 1:
        sys.exit(f"Yes/No 가 단일 토큰이 아니다: {yid} {nid}")
    YID, NID = yid[0], nid[0]
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    t0, n = time.time(), 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), a.chunk):
            batch = todo[s : s + a.chunk]
            prompts, index = [], []
            for i, outs in batch:
                for j, t in enumerate(outs):
                    prompts.append(build(tok, str(q.get(i, "")), t, a.stance))
                    index.append((i, j))
            res = llm.generate(prompts, sp)
            acc = {i: [None] * len(o) for i, o in batch}
            for (i, j), r in zip(index, res):
                lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
                y = lp[YID].logprob if YID in lp else FLOOR
                nn = lp[NID].logprob if NID in lp else FLOOR
                acc[i][j] = round(y - nn, 4)
            for i, o in batch:
                f.write(json.dumps({"id": i, "orm": acc[i]}, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
            n += len(prompts)
            el = time.time() - t0
            print(f"[orm] {s+len(batch)}/{len(todo)} 문항 · {n:,} 채점 · "
                  f"{n/max(el,1e-9):.0f} 건/s · 경과 {el/60:.1f}분", flush=True)

    print(f"저장: {out}")


if __name__ == "__main__":
    main()
