"""격리 판정을 다시 매긴다 — 이미 만든 gen_quar_*.jsonl 로, GPU 없이.

만장일치(16/16)는 어려운 문항을 전부 drop 시킨다. 실측에서 편입률이 19.8%로
사전등록 예측(30% 이상)을 밑돌았다. 기준을 낮추면 얼마나 회수되는지,
그리고 그 대가로 어떤 위험이 늘어나는지를 여기서 본다.

  python3 src/requarantine.py --gen out/gen_quar_r1.jsonl \
      --quarantine out/train_quarantine.csv
"""
import argparse, collections, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", default="out/gen_quar_r1.jsonl")
    ap.add_argument("--quarantine", default="out/train_quarantine.csv")
    ap.add_argument("--out", default="out/requarantine_report.json")
    a = ap.parse_args()

    q = pd.read_csv(a.quarantine)
    cur = dict(zip(q["id"], pd.to_numeric(q["current_label"], errors="coerce")))
    sug = dict(zip(q["id"], pd.to_numeric(q["suggested_answer"], errors="coerce")))

    preds = {}
    with open(a.gen, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            preds[r["id"]] = [extract_answer(t) for t in r["outputs"]]

    ids = [i for i in q["id"] if i in preds]
    print(f"격리 {len(ids)}문항 · 샘플 {len(preds[ids[0]])}개\n")
    print(f"{'합의 기준':>10} {'relabel':>9} {'keep':>7} {'drop':>7} {'편입률':>8}")
    print("-" * 46)

    rows = {}
    for th in (1.00, 0.95, 0.90, 0.85, 0.75, 0.60, 0.50):
        v = collections.Counter()
        for i in ids:
            p = preds[i]
            nn = [x for x in p if x is not None]
            top = None
            if nn:
                t, c = collections.Counter(nn).most_common(1)[0]
                if c / len(p) >= th:
                    top = t
            s, cu = sug.get(i), cur.get(i)
            if top is not None and s == s and top == int(s):
                v["relabel"] += 1
            elif top is not None and cu == cu and top == int(cu):
                v["keep"] += 1
            else:
                v["drop"] += 1
        rate = (v["relabel"] + v["keep"]) / len(ids)
        rows[f"{th:.2f}"] = {**v, "rate": round(rate, 4)}
        mark = "  ← 현재" if th == 1.00 else ("  ← 예측 30% 통과" if rate >= .30 else "")
        print(f"{th:>10.2f} {v['relabel']:>9} {v['keep']:>7} {v['drop']:>7} {rate:>7.1%}{mark}")

    # 위험 지표: 모델 최빈답이 제안값도 기존값도 아닌 제3의 값인 경우
    third = sum(1 for i in ids
                if (nn := [x for x in preds[i] if x is not None])
                and (t := collections.Counter(nn).most_common(1)[0][0]) is not None
                and t != (int(sug[i]) if sug.get(i) == sug.get(i) else None)
                and t != (int(cur[i]) if cur.get(i) == cur.get(i) else None))
    print(f"\n모델 최빈답이 제3의 값: {third}/{len(ids)} ({third/len(ids):.1%})")
    print("  이 문항들은 기준을 낮춰도 채택되지 않는다 (제안·기존 어느 쪽과도 불일치).")
    print("  기준 완화로 회수되는 것은 '합의는 있으나 만장일치는 아니었던' 문항뿐이다.")

    Path(a.out).write_text(json.dumps(
        {"n": len(ids), "by_threshold": rows, "third_value": third},
        ensure_ascii=False, indent=2))
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
