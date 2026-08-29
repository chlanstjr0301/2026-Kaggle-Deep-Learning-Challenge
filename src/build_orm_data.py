"""검증기(ORM) 학습 데이터를 RFT 산출물에서 만든다. 추가 생성 없음.

설계에서 가장 중요한 결정 하나: **문항 간이 아니라 문항 안에서 가르친다.**

RFT 데이터를 그대로 쓰면 정답 풀이는 쉬운 문항에, 오답 풀이는 어려운 문항에
쏠려 있다. 그걸로 학습하면 검증기가 "이 문제는 어려워 보인다 -> 오답"을 배운다.
전체 AUC는 높게 나오지만 **정작 필요한 능력(같은 문항의 여러 후보 중 고르기)은
전혀 늘지 않는다.** 그래서:

  - 정답과 오답이 **둘 다 있는 문항만** 사용한다 (전부 맞거나 전부 틀린 문항 제외)
  - 문항마다 정답/오답을 **같은 개수로** 뽑는다
  - 라벨 의심 문항은 제외한다 (오답 라벨이 오염 신호가 된다)

  python3 src/build_orm_data.py --gen out/gen_rft_r1.jsonl --data out/train_clean.csv
"""
import argparse, collections, json, random, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer

MIN_CHARS = 100


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", default="out/gen_rft_r1.jsonl")
    ap.add_argument("--data", default="out/train_clean.csv")
    ap.add_argument("--suspect", default="out/label_suspect_r1.csv")
    ap.add_argument("--out", default="out/orm_r1.jsonl")
    ap.add_argument("--per-side", type=int, default=3,
                    help="문항당 정답·오답 각각 최대 몇 개")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if not Path(a.gen).exists():
        cands = sorted(Path("out").glob("gen_rft*.jsonl"))
        sys.exit(f"{a.gen} 없음. 후보: {[c.name for c in cands]}")

    df = pd.read_csv(a.data)
    gold = dict(zip(df["id"], df["answer"]))
    qtext = dict(zip(df["id"], df["question"]))
    bad = set()
    if Path(a.suspect).exists():
        try:
            bad = set(pd.read_csv(a.suspect)["id"])
        except Exception:
            pass

    rng = random.Random(a.seed)
    rows, stat = [], collections.Counter()
    n_prob = n_mixed = 0

    with open(a.gen, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            i = r["id"]
            if i not in gold:
                continue
            n_prob += 1
            if i in bad:
                stat["라벨의심 제외"] += 1
                continue
            texts = r["outputs"]
            tr = r.get("trunc", [False] * len(texts))
            g = int(gold[i])
            pos, neg = [], []
            for t, cut in zip(texts, tr):
                if cut or len(t) < MIN_CHARS:
                    continue
                v = extract_answer(t)
                if v is None:
                    continue                       # 형식 실패는 검증기 학습에 잡음
                (pos if v == g else neg).append(t)
            if not pos or not neg:
                stat["전부정답 또는 전부오답 제외"] += 1
                continue
            n_mixed += 1
            # 문항 내 균형: 적은 쪽에 맞춘다
            m = min(len(pos), len(neg), a.per_side)
            # 다수결이 틀린 문항인가 (검증기가 실제로 값을 하는 지점)
            cnt = collections.Counter()
            for t in texts:
                v = extract_answer(t)
                if v is not None:
                    cnt[v] += 1
            maj_wrong = bool(cnt) and cnt.most_common(1)[0][0] != g
            for t in rng.sample(pos, m):
                rows.append({"id": i, "question": qtext[i], "solution": t,
                             "label": 1, "maj_wrong": maj_wrong})
            for t in rng.sample(neg, m):
                rows.append({"id": i, "question": qtext[i], "solution": t,
                             "label": 0, "maj_wrong": maj_wrong})
            stat["사용 문항"] += 1

    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mw = sum(1 for r in rows if r["maj_wrong"])
    print(f"전체 문항 {n_prob:,} · 혼합 문항 {n_mixed:,} ({n_mixed/max(n_prob,1):.1%})")
    for k, v in stat.most_common():
        print(f"  {k:<28} {v:,}")
    print(f"\n표본 {len(rows):,}  (정답 {sum(r['label'] for r in rows):,} / "
          f"오답 {len(rows)-sum(r['label'] for r in rows):,})")
    print(f"다수결이 틀린 문항에서 온 표본 {mw:,} ({mw/max(len(rows),1):.1%})"
          f"  ← 검증기가 실제로 값을 하는 지점")
    print(f"저장: {a.out}")
    if n_mixed < 2000:
        print("\n!! 혼합 문항이 적다. per-side 를 늘리거나 k 를 키운 RFT 라운드가 필요하다.")


if __name__ == "__main__":
    main()
