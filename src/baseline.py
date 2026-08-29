"""D-9: 베이스라인 측정.

목적은 점수가 아니라 (1) 프롬프트 선택 (2) tok/s 실측 (3) val<->LB 상관 확인.
실측 tok/s가 나오면 전체 예산표를 다시 계산해야 한다.

  python src/baseline.py --model ./qwen3b --val out/val.csv
  python src/baseline.py --model ./qwen3b --val data/deep_chal_math_leaderboard_filtered.csv \
                         --no-answer --k 8 --temp 0.8 --submit out/submission_d9.csv
"""
import argparse, json, sys, re, collections
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer, extract_or_fallback, _find_boxed
from prompts import PROMPTS, build, build_prompt
from gen import generate


def model_tag(path: str) -> str:
    """모델 식별자. 캐시 키에 반드시 들어가야 한다 —— 이게 없으면 SFT 모델 평가가
    베이스 모델의 생성 결과를 조용히 재사용하고 '완료'라고 보고한다."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]", "_", Path(path).name.strip("/")) or "model"


def diagnose(text: str, trunc: bool, gold, got):
    """오답 1건이 어느 단계에서 깨졌는지 분류. 처방이 단계마다 다르다."""
    if trunc:
        return "잘림"                     # -> max_tokens를 올린다
    b = _find_boxed(text)
    if b is None:
        return "boxed없음"                # -> 프롬프트로 형식을 강제한다
    if got is None:
        return "boxed파싱실패"            # -> extract.py를 고친다
    return "모델오답"                     # -> 학습으로만 고칠 수 있다


def majority(preds):
    c = collections.Counter(p for p in preds if p is not None)
    return c.most_common(1)[0][0] if c else None


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--prompts", default="A,B,C")
    ap.add_argument("--modes", default="chat,raw",
                    help="chat=Instruct chat template 적용 / raw=평문. D-9에서 실측 비교")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N문항만 (0=전체)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=3072)   # D-9 실측으로 확정
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--logprobs", action="store_true",
                    help="샘플 토큰 logprob을 받는다. 이걸 켜야 vLLM이 cumulative_logprob을\n                         채운다 —— 끄면 gen.py의 logp가 전부 null이 되고 로그확률 집계\n                         규칙이 조용히 단순 다수결로 떨어진다 (2026-08-24 E15).")
    ap.add_argument("--no-answer", action="store_true", help="정답 없는 집합(리더보드/테스트)")
    ap.add_argument("--submit", default=None, help="제출 파일 경로")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    df = pd.read_csv(a.val)
    if a.limit:
        df = df.head(a.limit)
    rows = list(zip(df["id"], df["question"]))
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=4096, seed=42)
    sp = SamplingParams(n=a.k, temperature=a.temp, top_p=0.95 if a.temp > 0 else 1.0,
                        max_tokens=a.max_tokens, seed=42,
                        logprobs=0 if a.logprobs else None)

    tok = llm.get_tokenizer()
    combos = [(k.strip(), m.strip())
              for m in a.modes.split(",")
              for k in a.prompts.split(",")
              if k.strip() in PROMPTS]

    report = {}
    for key, mode in combos:
        name = f"{key}-{mode}"
        tag = (f"{model_tag(a.model)}__{Path(a.val).stem}_{key}_{mode}"
               f"_k{a.k}_t{a.temp}_mt{a.max_tokens}"
               + ("_lp" if a.logprobs else ""))
        pf = ((lambda q, k=key: build_prompt(tok, q, k, chat=True)) if mode == "chat"
              else (lambda q, k=key: build(q, k)))
        outs, perf = generate(llm, sp, rows, od / f"gen_{tag}.jsonl", pf)
        trmap = perf.pop("_trunc", {}); perf.pop("_logp", None)

        preds = {i: [extract_answer(t) for t in outs.get(i, [])] for i, _ in rows}
        final = {i: majority(p) for i, p in preds.items()}

        entry = {"perf": perf,
                 "unparsed_ratio": round(sum(v is None for v in final.values()) / len(final), 4),
                 "mean_out_chars": round(
                     sum(len(t) for v in outs.values() for t in v) /
                     max(sum(len(v) for v in outs.values()), 1), 1)}

        if not a.no_answer:
            gold = dict(zip(df["id"], df["answer"]))
            entry["acc"] = round(sum(final[i] == gold[i] for i in final) / len(final), 4)
            n = len(final)
            # 95% 신뢰구간 반폭 — 이보다 작은 칸 간 차이는 우연이다
            entry["acc_ci95"] = round(1.96 * (entry["acc"] * (1 - entry["acc"]) / n) ** .5, 4)

            # 오답을 단계별로 분해한다. 처방이 각각 다르므로 합계보다 이 분포가 중요하다.
            cause = collections.Counter()
            for i, _ in rows:
                if final[i] == gold[i]:
                    continue
                texts = outs.get(i, []) or [""]
                tr = trmap.get(i, [False] * len(texts))
                if all(tr):                      # 전부 잘렸을 때만 '잘림'
                    cause["잘림"] += 1
                    continue
                # 잘리지 않은 첫 샘플로 원인을 판정한다 (다수결의 대표 경로)
                j = next((x for x, t in enumerate(tr) if not t), 0)
                pv = preds[i][j] if preds.get(i) and j < len(preds[i]) else None
                cause[diagnose(texts[j], False, gold[i], pv)] += 1
            entry["오답분해"] = {k: round(v / n, 4) for k, v in cause.most_common()}

            # 정답 숫자가 본문에 독립 토큰으로 등장했는데 최종 추출이 놓친 비율.
            # (느슨한 지표 — 작은 정수는 우연히 등장할 수 있으니 상한으로만 읽을 것)
            miss = sum(1 for i, _ in rows
                       if final[i] != gold[i]
                       and any(re.search(rf"(?<![\d.]){re.escape(str(gold[i]))}(?![\d.])", t)
                               for t in outs.get(i, [])))
            entry["miss_upper"] = round(miss / n, 4)
            if a.k > 1:
                entry["pass_at_k"] = round(
                    sum(any(p == gold[i] for p in preds[i]) for i in preds) / len(preds), 4)
        report[name] = entry
        print(f"\n[{name}] {json.dumps(entry, ensure_ascii=False)}\n")

        if a.submit and (key, mode) == combos[-1]:
            n_fb = 0
            ans = []
            for i in df["id"]:
                if final[i] is not None:
                    ans.append(int(final[i]))
                else:
                    n_fb += 1
                    texts = outs.get(i, [])
                    ans.append(extract_or_fallback(texts[0] if texts else ""))
            # Kaggle 채점기는 소문자 "id"를 요구한다. Overview 문서에는 "ID"로
            # 적혀 있지만 실제 채점기가 기준이다 (2026-08-23 제출 오류로 확인).
            sub = pd.DataFrame({"id": df["id"], "answer": ans})
            sub["answer"] = sub["answer"].astype(int)
            sub.to_csv(a.submit, index=False)
            print(f"제출 파일 저장: {a.submit} ({len(sub)}행, 폴백 {n_fb}개 = {n_fb/len(sub):.1%})")

    (od / "baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
