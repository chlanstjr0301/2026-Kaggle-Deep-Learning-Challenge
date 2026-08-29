"""D-8 병목 진단. k개 샘플이 담긴 gen_*.jsonl 하나로 다음을 계산한다.

  pass@1        한 번 뽑아 맞을 확률 (모델의 소양)
  maj@m         m개 다수결 정확도 (self-consistency가 실제로 주는 이득)
  pass@m        m개 중 하나라도 맞을 확률 (탐색의 상한)
  pass@k - maj@k   "맞출 줄은 아는데 못 고르는" 간격 = 검증기/재순위의 여지

해석:
  pass@k가 낮다   -> 모델이 못 푼다. 학습(RFT)에 자원을 쏟아야 한다.
  pass@k는 높은데 maj@k가 낮다 -> 고르는 문제. 검증기·가중투표가 이득이다.
  maj@m이 일찍 평평해진다 -> k를 더 키워도 소용없다. 다른 데 쓴다.

  python3 src/diag.py --gen out/gen_val_C_chat_k64_t0.8_mt3072.jsonl --val out/val.csv --limit 300
"""
import argparse, json, math, random, sys, collections
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer

TRIVIAL = {0, 1}          # 합의 조기중단에서 제외하는 값


def unbiased_pass_at_m(n, c, m):
    """n개 중 c개가 정답일 때 m개 뽑아 하나라도 맞을 확률 (Codex 논문 추정량)."""
    if n - c < m:
        return 1.0
    return 1.0 - math.comb(n - c, m) / math.comb(n, m)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drop-trunc", action="store_true",
                    help="잘린 샘플을 투표에서 제외 (기본은 포함)")
    ap.add_argument("--trials", type=int, default=200, help="maj@m 부트스트랩 반복")
    ap.add_argument("--out", default="out/diag_report.json")
    a = ap.parse_args()

    df = pd.read_csv(a.val)
    if a.limit:
        df = df.head(a.limit)
    gold = dict(zip(df["id"], df["answer"]))

    preds, K = {}, 0
    with open(a.gen, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["id"] not in gold:
                continue
            texts = r["outputs"]
            tr = r.get("trunc", [False] * len(texts))
            p = [extract_answer(t) for t in texts]
            if a.drop_trunc:
                p = [v for v, t in zip(p, tr) if not t]
            preds[r["id"]] = p
            K = max(K, len(texts))

    ids = [i for i in df["id"] if i in preds]
    if not ids:
        sys.exit("겹치는 id 없음")
    n = len(ids)
    ms = [m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if m <= K]
    rng = random.Random(42)

    rows = []
    for m in ms:
        # maj@m: 복원 없이 m개를 뽑아 다수결. trials회 평균.
        hit = 0.0
        for i in ids:
            p = preds[i]
            if not p:
                continue
            ok = 0
            for _ in range(a.trials):
                s = rng.sample(p, m) if len(p) >= m else [rng.choice(p) for _ in range(m)]
                c = collections.Counter(v for v in s if v is not None)
                ok += (c.most_common(1)[0][0] == gold[i]) if c else 0
            hit += ok / a.trials
        maj = hit / n
        # pass@m: 편향 없는 추정량
        pm = sum(unbiased_pass_at_m(len(preds[i]), sum(v == gold[i] for v in preds[i]), m)
                 for i in ids if preds[i]) / n
        rows.append((m, maj, pm))

    print(f"파일: {Path(a.gen).name}   문항 {n}   샘플 k={K}"
          + ("   (잘린 샘플 투표 제외)" if a.drop_trunc else ""))
    print(f"\n{'m':>4} {'maj@m':>8} {'pass@m':>8} {'격차':>8}")
    print("-" * 32)
    for m, maj, pm in rows:
        print(f"{m:>4} {maj:>8.4f} {pm:>8.4f} {pm-maj:>8.4f}")

    # 파싱 실패율 / 다양성
    flat = [v for i in ids for v in preds[i]]
    none_r = sum(v is None for v in flat) / max(len(flat), 1)
    uniq = sum(len({v for v in preds[i] if v is not None}) for i in ids) / n
    # 만장일치(합의 조기중단으로 아낄 수 있는 문항) 비율
    unan = sum(1 for i in ids
               if len({v for v in preds[i] if v is not None}) == 1
               and next(iter({v for v in preds[i] if v is not None}), None) not in TRIVIAL) / n
    print(f"\n샘플 파싱실패 {none_r:.2%}  ·  문항당 서로 다른 답 평균 {uniq:.2f}개"
          f"  ·  만장일치(비자명) {unan:.1%}")

    m_last, maj_k, pass_k = rows[-1]      # K가 아니라 실제로 계산한 m 을 쓴다
    print(f"\n판정:")
    print(f"  pass@{m_last} = {pass_k:.4f}  ← 이 모델이 지금 도달 가능한 상한")
    print(f"  maj@{m_last}  = {maj_k:.4f}  ← 다수결로 실제로 건지는 값")
    print(f"  격차      = {pass_k-maj_k:.4f}  ← 검증기·가중투표가 노릴 수 있는 최대치")
    if rows[-1][1] - rows[-2][1] < 0.005:
        print(f"  maj가 m={rows[-2][0]}→{K}에서 +{rows[-1][1]-rows[-2][1]:.4f}만 늘었다."
              f" k를 더 키우는 건 낭비다.")
    if pass_k - maj_k > 0.08:
        print("  격차가 크다 -> 검증기/재순위에 투자할 가치가 있다.")
    else:
        print("  격차가 작다 -> 고르는 문제가 아니라 푸는 문제. RFT/학습에 투자한다.")

    Path(a.out).write_text(json.dumps(
        {"file": Path(a.gen).name, "n": n, "k": K, "drop_trunc": a.drop_trunc,
         "curve": [{"m": m, "maj": round(maj, 4), "pass": round(pm, 4)} for m, maj, pm in rows],
         "none_ratio": round(none_r, 4), "uniq_answers": round(uniq, 2),
         "unanimous_nontrivial": round(unan, 4)}, ensure_ascii=False, indent=2))
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
