"""자아비판형 검증기 (critique → judge). 학습 없이 프롬프트만으로.

ORM r1 의 실패 지점은 명확했다: 다수결이 틀린 문항에서 AUC 0.5871.
문제와 풀이를 보고 **토큰 하나**로 즉시 판정하니, 모델이 체계적으로 속는
문제에서는 검증기도 같이 속는다. 검산할 시간을 안 준 것이다.

여기서는 판정 전에 비판을 먼저 **생성**시킨다:

  system : (harsh) 모든 풀이에 오류가 있다고 가정하고 시작하라
  user   : 문제 + 풀이 + "단계별로 검산하고 틀린 곳을 지적하라"
  assist : <비판문 생성>                                  ← 여기가 핵심
  user   : "최종 답이 맞나? Yes/No 한 단어로"
  assist : → 첫 토큰에서 logP(Yes) − logP(No)

--rounds 2 면 비판을 한 번 더 시킨다 ("놓친 것이 있는지 다시 보라").

출력 형식은 score_orm.py 와 동일하므로 fuse.py 가 그대로 읽는다.

  python3 src/score_orm_cot.py --model ./qwen3b \
      --gen out/gen_qwen3b__val_C_chat_k32_t0.8_mt3072_lp.jsonl --limit 300 \
      --out out/orm__cot_zs.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from orm_prompt import STANCES, DEFAULT_STANCE, YES, NO
from extract import extract_answer

FLOOR = -20.0

CRITIQUE = ("Problem:\n{q}\n\nProposed solution:\n{s}\n\n"
            "Check this solution step by step. Recompute each arithmetic step "
            "yourself. State any error you find, or state that each step checks out. "
            "Be brief — at most 5 sentences.")

# 역방향 검증 —— 풀이를 읽지 않는다. 답만 받아 문제 조건에 대입한다.
# critique 모드가 실패한 이유가 "제시된 풀이에 동조"였으므로(전부 checks out),
# 앵커를 아예 제거한다. 순방향 풀이와 계산 경로가 달라 오차 상관이 깨질 여지가 있다.
BACKWARD = ("Problem:\n{q}\n\nA candidate final answer is: {a}\n\n"
            "Do NOT try to solve the problem from scratch, and do NOT assume the "
            "candidate is right. Instead, substitute {a} back into the problem and "
            "test it against each condition the problem states. List the conditions "
            "one by one and say whether {a} satisfies each. Be brief.")
BACK_VERDICT = ("Based only on your substitution check, is {a} consistent with every "
                "condition in the problem? Answer with exactly one word: Yes or No.")
RECHECK = ("Re-examine your own check. Did you miss anything — a misread quantity, "
           "a wrong operation, an unjustified assumption? Be brief.")
VERDICT = ("Given your check, is the final answer of the proposed solution correct? "
           "Answer with exactly one word: Yes or No.")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model", required=True, help="베이스 또는 병합된 어댑터")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stance", default=DEFAULT_STANCE, choices=list(STANCES))
    ap.add_argument("--rounds", type=int, default=1, help="비판 라운드 수 (1 또는 2)")
    ap.add_argument("--mode", default="critique", choices=["critique", "backward"],
                    help="critique=풀이를 검산 / backward=답만 받아 문제 조건에 대입")
    ap.add_argument("--crit-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-ans-chars", type=int, default=20,
                    help="후보 답 문자열 상한. 반복 붕괴가 만든 수천 자리 정수가\n                         프롬프트를 폭파시킨다 (2026-08-25 E18)")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--save-critiques", default=None, help="비판문 표본 저장 경로")
    ap.add_argument("--out", required=True,
                    help="반드시 지정할 것 —— stance/rounds 가 자동 파일명에 안 들어간다")
    a = ap.parse_args()

    keep = None
    if a.limit:
        keep = set(pd.read_csv(a.val).head(a.limit)["id"])
    q = dict(zip(*[pd.read_csv(a.val)[c] for c in ("id", "question")]))

    rows = []
    for line in open(a.gen, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if keep is not None and r["id"] not in keep:
            continue
        rows.append((r["id"], r["outputs"]))

    done = set()
    if Path(a.out).exists():
        for line in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [x for x in rows if x[0] not in done]
    print(f"[cot] 전체 {len(rows)} · 완료 {len(done)} · 남음 {len(todo)} "
          f"· mode={a.mode} · stance={a.stance} · rounds={a.rounds}")
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_len, seed=42, enable_prefix_caching=True)
    tok = llm.get_tokenizer()

    sp_crit = SamplingParams(n=1, temperature=0.0, max_tokens=a.crit_tokens, seed=42)
    sp_judge = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=20, seed=42)

    yid = tok.encode(YES, add_special_tokens=False)
    nid = tok.encode(NO, add_special_tokens=False)
    if len(yid) != 1 or len(nid) != 1:
        sys.exit(f"Yes/No 가 단일 토큰이 아니다: {yid} {nid} —— 점수 해석이 깨진다")
    yid, nid = yid[0], nid[0]

    f = open(a.out, "a", encoding="utf-8")
    fc = open(a.save_critiques, "a", encoding="utf-8") if a.save_critiques else None
    t0, n_scored = time.time(), 0

    for s in range(0, len(todo), a.chunk):
        batch = todo[s:s + a.chunk]
        # ── 각 후보마다 대화를 하나씩 만든다 ──────────────────────────
        convs, index, verd = [], [], []
        for i, texts in batch:
            for j, t in enumerate(texts):
                if a.mode == "backward":
                    ans = extract_answer(t)
                    astr = "(no parsable answer)" if ans is None else str(ans)
                    # 반복 붕괴로 나온 수천 자리 정수를 그대로 넣으면 프롬프트가
                    # 컨텍스트를 넘겨 배치 전체가 죽는다. 그런 답은 어차피 오답이므로
                    # 자리표시자로 바꾸고 점수는 자연히 낮게 나오게 둔다.
                    if len(astr) > a.max_ans_chars:
                        astr = "(implausibly long number, %d digits)" % len(astr)
                    convs.append([{"role": "system", "content": STANCES[a.stance]},
                                  {"role": "user",
                                   "content": BACKWARD.format(q=str(q.get(i, "")), a=astr)}])
                    verd.append(BACK_VERDICT.format(a=astr))
                else:
                    convs.append([{"role": "system", "content": STANCES[a.stance]},
                                  {"role": "user",
                                   "content": CRITIQUE.format(q=str(q.get(i, "")), s=t)}])
                    verd.append(VERDICT)
                index.append((i, j))

        # ── 컨텍스트 초과 항목을 미리 걸러낸다 (E18) ────────────────────
        budget = a.max_len - a.crit_tokens - 64
        keep, over = [], []
        for k_, c in enumerate(convs):
            n = len(tok.apply_chat_template(c, tokenize=True, add_generation_prompt=True))
            (keep if n <= budget else over).append(k_)
        if over:
            print(f"[cot] 컨텍스트 초과로 제외 {len(over)}건 (FLOOR 처리)", flush=True)
        convs_all, index_all, verd_all = convs, index, verd
        convs = [convs_all[k_] for k_ in keep]
        index = [index_all[k_] for k_ in keep]
        verd  = [verd_all[k_]  for k_ in keep]

        # ── 라운드별 비판 생성 ──────────────────────────────────────
        for rd in range(a.rounds):
            prompts = [tok.apply_chat_template(c, tokenize=False,
                                               add_generation_prompt=True) for c in convs]
            outs = llm.generate(prompts, sp_crit)
            for c, o in zip(convs, outs):
                c.append({"role": "assistant", "content": o.outputs[0].text.strip()})
                if rd + 1 < a.rounds:
                    c.append({"role": "user", "content": RECHECK})

        # ── 판정 ────────────────────────────────────────────────────
        for c, v in zip(convs, verd):
            c.append({"role": "user", "content": v})
        prompts = [tok.apply_chat_template(c, tokenize=False,
                                           add_generation_prompt=True) for c in convs]
        outs = llm.generate(prompts, sp_judge)

        score = {}
        for k_ in over:                       # 제외된 항목은 최저점
            i_, j_ = index_all[k_]
            score.setdefault(i_, {})[j_] = FLOOR
        for (i, j), o, c in zip(index, outs, convs):
            lp = (o.outputs[0].logprobs or [{}])[0] or {}
            gy = lp.get(yid); gn = lp.get(nid)
            y = gy.logprob if gy is not None else FLOOR
            n = gn.logprob if gn is not None else FLOOR
            score.setdefault(i, {})[j] = y - n
            if fc is not None and j == 0:
                fc.write(json.dumps({"id": i, "critique": c[2]["content"][:1200],
                                     "score": y - n}, ensure_ascii=False) + "\n")

        for i, texts in batch:
            d = score.get(i, {})
            f.write(json.dumps({"id": i, "orm": [d.get(j) for j in range(len(texts))]},
                               ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
        if fc is not None:
            fc.flush()
        n_scored += len(index)
        el = time.time() - t0
        print(f"[cot] {s+len(batch)}/{len(todo)} 문항 · {n_scored:,} 채점 "
              f"· {n_scored/max(el,1):.0f} 건/s · 경과 {el/60:.1f}분", flush=True)

    f.close()
    if fc is not None:
        fc.close()
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
