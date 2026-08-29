"""억까 재작성 —— 채점이 아니라 생성. pass@k 를 올리는 것이 목적이다.

팟캐스트의 원래 메커니즘은 "비판 -> 자아비판 -> 재작성 -> 반복"이고, 우리가 앞서
구현한 것은 그 중 '판정'만 떼어낸 것이었다(채점기 stance). 여기서는 재작성까지 간다.

왜 이것이 필요한가 —— 우리 실측이 말한다:
    pass@64 = 0.9367,  pass@128 = 0.9415   (k 를 2배로 늘려 +0.5%p)
pass@k 가 포화했다는 것은 온도 표집이 **같은 모드 안에서만** 돌고 있다는 뜻이다.
정답이 64번 중 한 번도 안 나오는 6.3% 문항은 k 를 키워서 못 고친다.
**강제 비판은 모드를 갈아타는 연산자다.** 온도가 못 하는 일을 한다.

핵심 성질: pass@k 는 후보를 추가해서 **내려갈 수 없다**. 재작성이 틀려도 상한은
안 떨어진다. 원본 풀은 투표용으로 그대로 두고, 재작성본은 검증기가 고를 수 있는
새 선택지로만 넣는다.

  python3 src/refine.py --model ./qwen3b \
      --gen out/gen_qwen3b__val_C_chat_k32_t0.8_mt3072_lp.jsonl \
      --limit 300 --per-problem 8 --rounds 1 --out out/refine_r1.jsonl
"""
import argparse, collections, json, os, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer

SYSTEM = "You are a careful mathematician. You always finish with a final integer answer."

# 억까 —— "확인해봐"가 아니라 "틀렸으니 어디가 틀렸는지 말해봐".
# 앞선 실험에서 "check this solution"은 전부 "Each step checks out"으로 끝났다.
# 오류를 찾지 못하면 재작성이 원본과 같아지고 모드를 못 벗어난다.
CRIT = ("Problem:\n{q}\n\nA student produced this solution:\n{s}\n\n"
        "This solution is wrong. Do not defend it. Identify the single most likely "
        "place where it goes wrong — a misread condition, a wrong operation, an "
        "arithmetic slip, or an unjustified assumption. Name it in one or two sentences.")
REVISE = ("Now solve the problem yourself from the start, avoiding that error. "
          "Show your steps. Put your final answer in \\boxed{}.")
AGAIN = ("Your new solution is also wrong. Find the most likely error in it, "
         "in one or two sentences.")


def pick_reps(texts, n):
    """서로 다른 답을 우선으로 대표 표본을 고른다 —— 같은 모드를 여러 번 때려봐야 소용없다."""
    by = collections.defaultdict(list)
    for j, t in enumerate(texts):
        by[extract_answer(t)].append(j)
    order = sorted(by.items(), key=lambda kv: -len(kv[1]))
    out, r = [], 0
    while len(out) < n and any(len(v) > r for _, v in order):
        for _, v in order:
            if len(v) > r and len(out) < n:
                out.append(v[r])
        r += 1
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-problem", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1, help="비판→재작성 반복 횟수")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--crit-tokens", type=int, default=192)
    ap.add_argument("--sol-tokens", type=int, default=1536)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.val)
    if a.limit:
        df = df.head(a.limit)
    q = dict(zip(df["id"], df["question"]))

    rows = []
    for line in open(a.gen, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["id"] in q:
            rows.append((r["id"], r["outputs"]))

    done = set()
    if Path(a.out).exists():
        for line in open(a.out, encoding="utf-8"):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    todo = [x for x in rows if x[0] not in done]
    print(f"[refine] 전체 {len(rows)} · 완료 {len(done)} · 남음 {len(todo)} "
          f"· 문항당 {a.per_problem}개 · rounds={a.rounds}", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_len, seed=42, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp_c = SamplingParams(n=1, temperature=a.temp, top_p=0.95, max_tokens=a.crit_tokens, seed=42)
    sp_s = SamplingParams(n=1, temperature=a.temp, top_p=0.95, max_tokens=a.sol_tokens, seed=42)

    f = open(a.out, "a", encoding="utf-8")
    t0, ntok = time.time(), 0
    for s in range(0, len(todo), a.chunk):
        batch = todo[s:s + a.chunk]
        convs, index = [], []
        for i, texts in batch:
            for j in pick_reps(texts, a.per_problem):
                convs.append([{"role": "system", "content": SYSTEM},
                              {"role": "user",
                               "content": CRIT.format(q=str(q[i]), s=texts[j])}])
                index.append((i, j))

        for rd in range(a.rounds):
            pr = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                  for c in convs]
            outs = llm.generate(pr, sp_c)                       # 비판
            for c, o in zip(convs, outs):
                c.append({"role": "assistant", "content": o.outputs[0].text.strip()})
                c.append({"role": "user", "content": REVISE})
                ntok += len(o.outputs[0].token_ids)
            pr = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                  for c in convs]
            outs = llm.generate(pr, sp_s)                       # 재작성
            for c, o in zip(convs, outs):
                c.append({"role": "assistant", "content": o.outputs[0].text.strip()})
                ntok += len(o.outputs[0].token_ids)
                if rd + 1 < a.rounds:
                    c.append({"role": "user", "content": AGAIN})

        res = collections.defaultdict(lambda: {"refined": [], "from": []})
        for (i, j), c in zip(index, convs):
            res[i]["refined"].append(c[-1]["content"])
            res[i]["from"].append(j)
        for i, _ in batch:
            d = res[i]
            f.write(json.dumps({"id": i, "refined": d["refined"], "from": d["from"]},
                               ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
        el = time.time() - t0
        print(f"[refine] {s+len(batch)}/{len(todo)} 문항 · {ntok:,} 토큰 "
              f"· {ntok/max(el,1):.0f} tok/s · {el/60:.1f}분", flush=True)
    f.close()
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
