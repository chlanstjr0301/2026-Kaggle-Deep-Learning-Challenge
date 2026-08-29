"""교사 생성물을 gold 로 걸러 SFT 데이터로 만든다. + 파일럿 진단 리포트.

  python3 src/teach_filter.py --gen out/teach_pilot.jsonl --report-only
  python3 src/teach_filter.py --gen out/teach_r1.jsonl --min-repeat 2 --max-per-problem 2 \
      --replay out/sft_r1.jsonl --replay-ratio 0.4 --out out/teach_sft.jsonl
"""
import argparse, collections, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer
import pandas as pd


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--data", default="out/train_clean.csv")
    ap.add_argument("--min-repeat", type=int, default=2,
                    help="정답이 이 횟수 이상 재현된 문항만 채택 (우연히 맞은 풀이 제거)")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=150)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--replay", default="out/sft_r1.jsonl", help="학생 자신의 정답 풀이")
    ap.add_argument("--replay-ratio", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    q = dict(zip(*[pd.read_csv(a.data)[c] for c in ("id", "question")]))
    rows = [json.loads(l) for l in open(a.gen, encoding="utf-8") if l.strip()]

    by = collections.defaultdict(lambda: {"n": 0, "any": 0, "rep": collections.Counter(),
                                          "chars": [], "ok_chars": []})
    kept, tok_hist = [], []
    for r in rows:
        b = r["bucket"]; g = r["answer"]
        s = by[b]; s["n"] += 1
        ok = []
        for t in r["outputs"]:
            s["chars"].append(len(t))
            if extract_answer(t) == g:
                ok.append(t)
        s["rep"][len(ok)] += 1
        if ok:
            s["any"] += 1
            s["ok_chars"] += [len(t) for t in ok]
        if len(ok) >= a.min_repeat:
            ok = [t for t in ok if a.min_chars <= len(t) <= a.max_chars]
            ok.sort(key=len)                       # 중앙 길이 우선
            mid = ok[len(ok)//2:] + ok[:len(ok)//2]
            for t in mid[:a.max_per_problem]:
                kept.append({"id": r["id"], "question": str(q.get(r["id"], "")),
                             "solution": t, "answer": g, "src": "teacher",
                             "bucket": b, "n_solved": r["n_solved"]})

    print(f"\n{'구간':<10}{'문항':>7}{'1회+정답':>10}{'해결률':>9}{'≥%d회' % a.min_repeat:>8}{'채택률':>9}")
    print("-" * 55)
    for b, s in by.items():
        mr = sum(v for k, v in s["rep"].items() if k >= a.min_repeat)
        print(f"{b:<10}{s['n']:>7,}{s['any']:>10,}{s['any']/max(s['n'],1):>9.1%}"
              f"{mr:>8,}{mr/max(s['n'],1):>9.1%}")
    print("-" * 55)
    for b, s in by.items():
        d = dict(sorted(s["rep"].items()))
        print(f"  {b} 정답 재현 분포: {d}")
        if s["ok_chars"]:
            c = sorted(s["ok_chars"])
            print(f"     정답 풀이 길이 중앙값 {c[len(c)//2]}자 "
                  f"(우리 모델 1,371자 · 95분위 {c[int(len(c)*.95)]}자)")

    print(f"\n채택 표본 {len(kept):,}건")
    if a.report_only or not a.out:
        print("(report-only —— 파일 안 씀)"); return

    rep = []
    if a.replay and Path(a.replay).exists() and a.replay_ratio > 0:
        pool = [json.loads(l) for l in open(a.replay, encoding="utf-8") if l.strip()]
        n = int(len(kept) * a.replay_ratio / max(1 - a.replay_ratio, 1e-9))
        random.Random(42).shuffle(pool)
        rep = [{**r, "src": "replay"} for r in pool[:n]]
        print(f"replay {len(rep):,}건 (비율 {len(rep)/max(len(kept)+len(rep),1):.0%}) "
              f"← 학생 자신의 풀이. SFT r1 이 무해했음이 실측돼 있어 앵커로 안전하다")

    allr = kept + rep
    random.Random(42).shuffle(allr)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {a.out}  (총 {len(allr):,}건)")


if __name__ == "__main__":
    main()
