"""억까 재작성의 효과를 pass@k 관점에서 판정한다.

판정 지표는 하나다: **원본 풀에 정답이 하나도 없던 문항 중 몇 개가 구제됐는가.**
전체 정확도가 아니라 이것이다 —— 재작성은 상한(pass)을 올리는 연산이고,
그 상한이 현재 우리 선별 노선 전체의 병목이기 때문이다.

  python3 src/refine_eval.py --gen out/gen_...jsonl --refined out/refine_r1.jsonl
"""
import argparse, collections, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()

    d = pd.read_csv(a.val).head(a.limit) if a.limit else pd.read_csv(a.val)
    gold = dict(zip(d["id"], d["answer"]))

    orig = {}
    for line in open(a.gen, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in gold:
            orig[r["id"]] = [extract_answer(t) for t in r["outputs"]]
    ref = {}
    for line in open(a.refined, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in gold:
            ref[r["id"]] = [extract_answer(t) for t in r["refined"]]

    ids = [i for i in d["id"] if i in orig and i in ref]
    print(f"문항 {len(ids)} · 원본 k={len(orig[ids[0]])} · 재작성 {len(ref[ids[0]])}\n")

    p0 = [i for i in ids if any(x == gold[i] for x in orig[i])]
    dead = [i for i in ids if i not in set(p0)]
    saved = [i for i in dead if any(x == gold[i] for x in ref[i])]

    newans = tot = changed = 0
    for i in ids:
        o = set(x for x in orig[i] if x is not None)
        for x in ref[i]:
            tot += 1
            if x is not None and x not in o:
                newans += 1
        maj = collections.Counter(x for x in orig[i] if x is not None)
        m = maj.most_common(1)[0][0] if maj else None
        changed += sum(1 for x in ref[i] if x is not None and x != m)

    print(f"원본 pass@{len(orig[ids[0]])}      {len(p0)}/{len(ids)} = {len(p0)/len(ids):.4f}")
    print(f"정답 0개였던 문항        {len(dead)}")
    print(f"  \033[1;32m그중 재작성이 구제      {len(saved)}  ({len(saved)/max(len(dead),1):.1%})\033[0m")
    union = len(p0) + len(saved)
    print(f"합집합 pass            {union}/{len(ids)} = {union/len(ids):.4f}  "
          f"({(union-len(p0))/len(ids):+.4f})")
    print(f"\n재작성이 원본 풀에 없던 답을 낸 비율   {newans}/{tot} = {newans/max(tot,1):.1%}")
    print(f"재작성이 다수결과 다른 답을 낸 비율     {changed}/{tot} = {changed/max(tot,1):.1%}")
    print("\n  위 두 비율이 낮으면 억까가 모드를 못 갈아탄 것이다 —— 프롬프트를 더 강하게.")
    print("  구제율이 0 이면 재작성 노선을 닫는다.")


if __name__ == "__main__":
    main()
