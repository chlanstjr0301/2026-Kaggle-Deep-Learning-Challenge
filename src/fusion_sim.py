r"""검증기를 학습하기 전에, 그게 얼마나 좋아야 값을 하는지 먼저 재는 시뮬레이션.

아이디어(칼만 필터의 정적 특수경우): 다수결과 검증기는 같은 미지수(정답)를 재는
서로 다른 계측기다. 하나를 버리고 다른 하나를 쓰는 게 아니라, 각자의 분산에
반비례하게 가중해서 합친다.

    score(a) = beta * log(n_a)  +  gamma * mean_verifier_score(a)
               \____ 다수결 ____/    \______ 검증기 ______/

    gamma/beta 가 칼만 게인에 해당한다. gamma=0 이면 순수 다수결,
    beta=0 이면 순수 Best-of-N. 최적점은 보통 그 사이에 있다.

검증기는 아직 없으므로, 품질을 AUC 로 매개변수화해 가짜 점수를 만든다.
정답 풀이에는 N(mu,1), 오답 풀이에는 N(0,1) 에서 점수를 뽑고
mu = sqrt(2)*Phi^-1(AUC) 로 두면 그 검증기의 AUC 가 정확히 목표값이 된다.

  python3 src/fusion_sim.py --gen out/gen_val_C_chat_k256_t0.8_mt3072.jsonl --m 32
"""
import argparse, json, math, random, sys, collections
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer


def _ppf(p):
    """표준정규 분위수 (Acklam 근사). scipy 없이 쓴다."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--out", default="out/fusion_sim.json")
    a = ap.parse_args()

    df = pd.read_csv(a.val).head(a.limit) if a.limit else pd.read_csv(a.val)
    gold = dict(zip(df["id"], df["answer"]))
    pool = {}
    with open(a.gen, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["id"] in gold:
                pool[r["id"]] = [extract_answer(t) for t in r["outputs"]]
    ids = [i for i in df["id"] if i in pool]
    m = min(a.m, min(len(v) for v in pool.values()))
    print(f"문항 {len(ids)} · 표본 m={m} · {a.trials}회 평균\n")

    AUCS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    GAMMAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]      # 1e9 = 사실상 순수 Best-of-N
    res = {}

    for auc in AUCS:
        mu = math.sqrt(2) * _ppf(min(max(auc, 1e-6), 1 - 1e-6))
        row = {}
        for g in GAMMAS:
            rng = random.Random(42)
            hit = 0.0
            for i in ids:
                ok = 0
                for _ in range(a.trials):
                    s = rng.sample(pool[i], m)
                    agg = collections.defaultdict(lambda: [0, 0.0])
                    for v in s:
                        sc = rng.gauss(mu if v == gold[i] else 0.0, 1.0)
                        if v is None:
                            continue
                        agg[v][0] += 1
                        agg[v][1] += sc
                    if not agg:
                        continue
                    if g >= 1e8:                       # 순수 Best-of-N: 최고 점수 1건
                        best = max(agg.items(), key=lambda kv: kv[1][1] / kv[1][0])[0]
                    else:
                        best = max(agg.items(),
                                   key=lambda kv: math.log(kv[1][0]) + g * (kv[1][1] / kv[1][0]))[0]
                    ok += (best == gold[i])
                hit += ok / a.trials
            row[g] = hit / len(ids)
        res[auc] = row

    base = res[0.50][0.0]
    hdr = "  ".join(f"{('BoN' if g>=1e8 else f'γ={g:g}'):>8}" for g in GAMMAS)
    print(f"{'검증기 AUC':>10}  {hdr}")
    print("-" * (12 + 10 * len(GAMMAS)))
    for auc in AUCS:
        cells = "  ".join(f"{res[auc][g]:>8.4f}" for g in GAMMAS)
        print(f"{auc:>10.2f}  {cells}")
    print("-" * (12 + 10 * len(GAMMAS)))
    print(f"γ=0 열이 순수 다수결({base:.4f}) · BoN 열이 순수 Best-of-N")

    print("\n검증기 품질별 최적 게인과 이득:")
    for auc in AUCS:
        g, v = max(res[auc].items(), key=lambda kv: kv[1])
        tag = "BoN" if g >= 1e8 else f"γ={g:g}"
        print(f"  AUC {auc:.2f} -> 최적 {tag:>6}  {v:.4f}  ({v-base:+.4f})")

    Path(a.out).write_text(json.dumps(
        {"n": len(ids), "m": m, "baseline_majority": round(base, 4),
         "grid": {str(k): {("bon" if g >= 1e8 else str(g)): round(x, 4)
                           for g, x in v.items()} for k, v in res.items()}},
        ensure_ascii=False, indent=2))
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
