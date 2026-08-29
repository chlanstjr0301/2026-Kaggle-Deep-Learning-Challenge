"""이미 생성된 gen_*.jsonl 들을 짝지어(paired) 비교한다. GPU 재생성 없음.

왜 짝비교인가: 6칸이 모두 "같은 300문항"을 풀었다. 독립표본 검정을 쓰면
문항 난이도 분산이 오차에 그대로 들어가 검정력을 버린다. 같은 문항에서
한쪽만 맞은 경우(불일치 쌍)만 세는 McNemar 검정이 정답이다.

  python3 src/analyze.py --val out/val.csv --limit 300 --glob 'out/gen_val_*_k1_t0.0.jsonl'
"""
import argparse, json, math, re, sys, collections
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer, _find_boxed


def binom_two_sided(b, c):
    """McNemar 정확검정. p = P(|X - n/2| >= |b - n/2|), X~Bin(n, .5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(path):
    out, tr = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["id"]] = r["outputs"]
            tr[r["id"]] = r.get("trunc", [False] * len(r["outputs"]))
    return out, tr


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--glob", default="out/gen_val_*_k1_t0.0.jsonl")
    ap.add_argument("--out", default="out/analyze_report.json")
    a = ap.parse_args()

    df = pd.read_csv(a.val)
    if a.limit:
        df = df.head(a.limit)
    gold = dict(zip(df["id"], df["answer"]))
    ids = list(df["id"])

    files = sorted(Path().glob(a.glob))
    if not files:
        sys.exit(f"파일 없음: {a.glob}")

    arms, report = {}, {}
    for fp in files:
        m = re.search(r"gen_.*?_([ABC])_(chat|raw)_k(\d+)_t([\d.]+?)(?:_mt(\d+))?\.jsonl$", fp.name)
        if m:
            name = f"{m.group(1)}-{m.group(2)}"
            if m.group(3) != "1":
                name += f"-k{m.group(3)}"
            name += f"-mt{m.group(5) or '1024?'}"     # 길이 예산이 다르면 다른 칸이다
        else:
            name = fp.stem
        if name in arms:
            sys.exit(f"칸 이름 충돌: {name} ({fp.name}). --glob을 좁혀라.")
        outs, tr = load(fp)
        correct, cause = {}, collections.Counter()
        for i in ids:
            texts = outs.get(i, []) or [""]
            got = extract_answer(texts[0])
            ok = (got == gold[i])
            correct[i] = ok
            if ok:
                continue
            if any(tr.get(i, [False])):        cause["잘림"] += 1
            elif _find_boxed(texts[0]) is None: cause["boxed없음"] += 1
            elif got is None:                   cause["boxed파싱실패"] += 1
            else:                               cause["모델오답"] += 1
        arms[name] = correct
        n = len(ids); acc = sum(correct.values()) / n
        report[name] = {"acc": round(acc, 4),
                        "n": n,
                        "오답분해": {k: v for k, v in cause.most_common()}}

    print(f"{'칸':<10} {'정확도':>8} {'잘림':>7} {'boxed없음':>10} {'파싱실패':>9} {'모델오답':>9}")
    print("-" * 60)
    for k, v in sorted(report.items(), key=lambda x: -x[1]["acc"]):
        c = v["오답분해"]
        print(f"{k:<10} {v['acc']:>8.4f} {c.get('잘림',0):>7} {c.get('boxed없음',0):>10} "
              f"{c.get('boxed파싱실패',0):>9} {c.get('모델오답',0):>9}")

    # --- 짝비교 (McNemar) ---
    names = sorted(arms)
    print(f"\n짝비교 (McNemar 정확검정) — b=앞만맞음, c=뒤만맞음")
    print(f"{'대비':<24} {'b':>4} {'c':>4} {'차이':>8} {'p':>8}")
    print("-" * 52)
    pairs = {}
    for x in names:
        for y in names:
            if x >= y:
                continue
            b = sum(1 for i in ids if arms[x][i] and not arms[y][i])
            c = sum(1 for i in ids if arms[y][i] and not arms[x][i])
            p = binom_two_sided(b, c)
            d = (b - c) / len(ids)
            pairs[f"{x} vs {y}"] = {"b": b, "c": c, "diff": round(d, 4), "p": round(p, 4)}
            star = "  <-- 유의" if p < 0.05 else ""
            print(f"{x+' vs '+y:<24} {b:>4} {c:>4} {d:>+8.4f} {p:>8.4f}{star}")

    # --- chat vs raw 를 프롬프트별로 묶어서 (같은 프롬프트끼리 짝) ---
    B = C = 0
    for key in "ABC":
        x, y = f"{key}-chat", f"{key}-raw"
        if x in arms and y in arms:
            B += sum(1 for i in ids if arms[x][i] and not arms[y][i])
            C += sum(1 for i in ids if arms[y][i] and not arms[x][i])
    if B + C:
        p = binom_two_sided(B, C)
        print(f"\nchat vs raw 통합 (A/B/C 3쌍, 900 짝): b={B} c={C} "
              f"차이={(B-C)/(3*len(ids)):+.4f} p={p:.5f}"
              f"{'  <-- 유의' if p < 0.05 else ''}")
        report["chat_vs_raw_pooled"] = {"b": B, "c": C, "p": round(p, 5)}

    # --- 상한: 어떤 칸이든 하나라도 맞춘 문항 비율 (프롬프트 앙상블 상한) ---
    any_ok = sum(1 for i in ids if any(arms[k][i] for k in names)) / len(ids)
    all_ok = sum(1 for i in ids if all(arms[k][i] for k in names)) / len(ids)
    print(f"\n합집합(어느 칸이든 정답) {any_ok:.4f}  ·  교집합(모든 칸 정답) {all_ok:.4f}")
    report["union_acc"] = round(any_ok, 4)
    report["intersect_acc"] = round(all_ok, 4)
    report["pairs"] = pairs

    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
