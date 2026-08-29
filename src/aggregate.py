"""집계 방식 비교. 이미 만든 gen_*.jsonl 하나로 돌린다 — GPU 재생성 없음.

배경: maj@m은 m=32에서 사실상 포화하는데 pass@m은 계속 오른다. 즉 정답 표가
안에 있는데 단순 다수결이 그걸 못 고른다. 격차가 m과 함께 벌어지므로,
표본을 더 뽑는 대신 **고르는 방법**을 바꾸는 쪽이 남은 여지다.

여기서 비교하는 것들은 전부 추가 학습 없이 쓸 수 있는 집계 규칙이다.

  python3 src/aggregate.py --gen out/gen_val_C_chat_k256_t0.8_mt3072.jsonl \
      --val out/val.csv --limit 300 --m 32
"""
import argparse, json, math, re, sys, collections, random
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer

_OPS = re.compile(r"-?\d[\d,]*(?:\.\d+)?\s*[-+*/×÷]\s*-?\d[\d,]*(?:\.\d+)?")


def path_hash(t):
    """계산 경로 해시. 같은 산술 순서열이면 같은 풀이로 본다."""
    ops = _OPS.findall(t)
    return "|".join(o.replace(" ", "").replace(",", "") for o in ops) or t[:80]


# ── 집계 규칙들 ────────────────────────────────────────────
# 각 규칙은 (답, 텍스트, 잘림여부) 목록을 받아 최종 정수를 낸다.

def agg_plain(rows):
    c = collections.Counter(r[0] for r in rows if r[0] is not None)
    return c.most_common(1)[0][0] if c else None


def agg_no_trunc(rows):
    """잘린 샘플은 표에서 뺀다. 잘린 풀이는 결론에 도달하지 못했다."""
    r = [x for x in rows if not x[2]] or rows
    return agg_plain(r)


def agg_path_unique(rows):
    """같은 계산 경로는 한 표로 친다.
    동일한 풀이를 반복해서 뽑은 것이 표를 부풀리는 것을 막는다."""
    seen, uniq = set(), []
    for r in rows:
        h = path_hash(r[1])
        if h in seen:
            continue
        seen.add(h); uniq.append(r)
    return agg_plain(uniq)


def agg_median_len(rows):
    """길이 중앙값에 가까운 풀이에 가중.
    너무 짧으면 추론 생략, 너무 길면 헤맨 것이라는 가정."""
    lens = sorted(len(r[1]) for r in rows)
    if not lens:
        return None
    med = lens[len(lens) // 2] or 1
    w = collections.Counter()
    for r in rows:
        if r[0] is None:
            continue
        w[r[0]] += 1.0 / (1.0 + abs(len(r[1]) - med) / med)
    return w.most_common(1)[0][0] if w else None


def agg_short(rows):
    """짧을수록 가중 — 대조군. 중앙값 선호가 실제로 나은지 확인하려는 것."""
    w = collections.Counter()
    for r in rows:
        if r[0] is None:
            continue
        w[r[0]] += 1.0 / (1.0 + len(r[1]) / 1000.0)
    return w.most_common(1)[0][0] if w else None


def agg_combo(rows):
    """잘림 제외 + 경로 중복 제거 + 중앙값 가중."""
    r = [x for x in rows if not x[2]] or rows
    seen, uniq = set(), []
    for x in r:
        h = path_hash(x[1])
        if h in seen:
            continue
        seen.add(h); uniq.append(x)
    return agg_median_len(uniq or r)


def _lp_weighted(rows, mode):
    """평균 로그확률로 가중. rows 원소는 (답, 텍스트, 잘림, 평균logp).
    logp 가 없으면 단순 다수결로 떨어진다."""
    lps = [r[3] for r in rows if len(r) > 3 and r[3] is not None]
    if not lps:
        return agg_plain([(r[0], r[1], r[2]) for r in rows])
    mu = sum(lps) / len(lps)
    sd = (sum((x - mu) ** 2 for x in lps) / max(len(lps) - 1, 1)) ** .5 or 1.0
    w = collections.Counter()
    for r in rows:
        a_, lp = r[0], (r[3] if len(r) > 3 else None)
        if a_ is None:
            continue
        z = 0.0 if lp is None else (lp - mu) / sd
        if mode == "soft":
            w[a_] += math.exp(min(max(z, -6), 6))      # 다수결 + 확신도 융합
        else:
            w[a_] = max(w[a_], z) if a_ in w else z    # 최고 확신 1건 (BoN)
    return w.most_common(1)[0][0] if w else None


def agg_logp_soft(rows):
    return _lp_weighted(rows, "soft")


def agg_logp_best(rows):
    return _lp_weighted(rows, "best")


AGGS = [("단순 다수결", agg_plain),
        ("로그확률 가중", agg_logp_soft),
        ("최고 로그확률(BoN)", agg_logp_best),
        ("잘림 제외", agg_no_trunc),
        ("경로 중복 제거", agg_path_unique),
        ("중앙길이 가중", agg_median_len),
        ("짧은길이 가중", agg_short),
        ("결합(잘림+경로+중앙)", agg_combo)]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--m", type=int, default=32, help="표본 수 (부분 표집)")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--out", default="out/aggregate_report.json")
    a = ap.parse_args()

    df = pd.read_csv(a.val)
    if a.limit:
        df = df.head(a.limit)
    gold = dict(zip(df["id"], df["answer"]))

    pool = {}
    with open(a.gen, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["id"] not in gold:
                continue
            ts = r["outputs"]
            tr = r.get("trunc", [False] * len(ts))
            lp = r.get("logp", [None] * len(ts))
            pool[r["id"]] = [(extract_answer(t), t, bool(x), y)
                             for t, x, y in zip(ts, tr, lp)]

    ids = [i for i in df["id"] if i in pool]
    K = max(len(v) for v in pool.values())
    m = min(a.m, K)
    rng = random.Random(42)
    print(f"파일 {Path(a.gen).name} · 문항 {len(ids)} · 풀 k={K} · 표본 m={m} · {a.trials}회 평균\n")

    # logp 가용성을 먼저 크게 찍는다. 전부 null이면 로그확률 규칙 두 개가 조용히
    # 단순 다수결로 떨어져 "Δ=+0.0000 = 효과 없음"처럼 보인다. 그건 신호 없음이
    # 아니라 데이터 없음이다 (2026-08-24 E15). 구분 못 하면 잘못된 결론을 내린다.
    tot = sum(len(v) for v in pool.values())
    have = sum(1 for v in pool.values() for x in v if len(x) > 3 and x[3] is not None)
    if have == 0:
        print("\033[1;31m[경고] logp 가 하나도 없다 (0/%d).\033[0m "
              "→ '로그확률 가중'·'최고 로그확률' 두 행은 단순 다수결의 복사본이다.\n"
              "        생성 때 --logprobs 를 켜야 vLLM 이 cumulative_logprob 을 채운다.\n"
              % tot)
    elif have < tot:
        print(f"\033[1;33m[참고] logp 결측 {tot-have}/{tot} ({1-have/tot:.1%}) "
              f"— 해당 샘플은 z=0 으로 처리된다.\033[0m\n")
    LOGP_OK = have > 0

    # 상한: m개 안에 정답이 하나라도 있으면 맞은 것으로 친 값
    oracle = 0.0
    for i in ids:
        hit = 0
        for _ in range(a.trials):
            s = rng.sample(pool[i], m)
            hit += any(x[0] == gold[i] for x in s)
        oracle += hit / a.trials
    oracle /= len(ids)

    res = {}
    for name, fn in AGGS:
        rng2 = random.Random(42)          # 모든 규칙이 같은 부분표본을 본다
        hit = 0.0
        for i in ids:
            ok = 0
            for _ in range(a.trials):
                s = rng2.sample(pool[i], m)
                ok += (fn(s) == gold[i])
            hit += ok / a.trials
        res[name] = hit / len(ids)

    base = res["단순 다수결"]
    print(f"{'집계 규칙':<24} {'정확도':>8} {'Δ':>9} {'격차 회수':>10}")
    print("-" * 56)
    for name, v in sorted(res.items(), key=lambda x: -x[1]):
        d = v - base
        rec = d / (oracle - base) if oracle > base else 0.0
        mark = "  ←" if name == "단순 다수결" else ""
        if not LOGP_OK and "로그확률" in name:
            mark = "  ← logp 없음: 무효"
        print(f"{name:<24} {v:>8.4f} {d:>+9.4f} {rec:>9.1%}{mark}")
    print("-" * 56)
    print(f"{'pass@m (상한)':<24} {oracle:>8.4f} {oracle-base:>+9.4f}")

    n = len(ids)
    se = math.sqrt(base * (1 - base) / n)
    print(f"\n참고: n={n}에서 절대 정확도 SE = {se:.4f}. "
          f"이보다 작은 Δ는 신뢰하지 말 것.")
    print("     (모든 규칙이 동일한 부분표본을 보므로 규칙 간 비교는 짝비교라 더 민감하다)")

    Path(a.out).write_text(json.dumps(
        {"file": Path(a.gen).name, "n": n, "k": K, "m": m, "trials": a.trials,
         "oracle": round(oracle, 4),
         "results": {k: round(v, 4) for k, v in res.items()}},
        ensure_ascii=False, indent=2))
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
