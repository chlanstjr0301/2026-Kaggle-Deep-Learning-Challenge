"""오답을 원인별로 꺼내 본다. 특히 'boxed파싱실패'는 extract.py를 고치면 바로 회수된다.

  python3 src/inspect_fail.py --gen out/gen_val_C_chat_k1_t0.0_mt1024.jsonl --cause 파싱실패
  python3 src/inspect_fail.py --gen out/gen_val_A_chat_k1_t0.0_mt1024.jsonl --cause 잘림 -n 3
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer, _find_boxed

CAUSES = ["잘림", "boxed없음", "파싱실패", "모델오답"]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--cause", default="파싱실패", choices=CAUSES + ["전체"])
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--chars", type=int, default=600, help="본문 끝에서 보여줄 길이")
    a = ap.parse_args()

    df = pd.read_csv(a.val)
    gold = dict(zip(df["id"], df["answer"]))
    ques = dict(zip(df["id"], df["question"]))

    shown = 0
    with open(a.gen, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            i = r["id"]
            if i not in gold:
                continue
            t = (r["outputs"] or [""])[0]
            tr = any(r.get("trunc", [False]))
            got = extract_answer(t)
            if got == gold[i]:
                continue
            b = _find_boxed(t)
            cause = ("잘림" if tr else "boxed없음" if b is None
                     else "파싱실패" if got is None else "모델오답")
            if a.cause != "전체" and cause != a.cause:
                continue
            shown += 1
            print("=" * 70)
            print(f"[{cause}] id={i}  정답={gold[i]}  추출={got}")
            print(f"문제: {str(ques[i])[:300]}")
            if b is not None:
                print(f"boxed 내용: {b!r}")
            print(f"--- 본문 끝 {a.chars}자 ---")
            print(t[-a.chars:])
            if shown >= a.n:
                break
    print("=" * 70)
    print(f"{a.cause}: {shown}건 출력")


if __name__ == "__main__":
    main()
