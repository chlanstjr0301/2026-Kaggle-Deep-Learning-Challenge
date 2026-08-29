"""다수결 + 검증기 융합으로 제출 파일을 만든다.

    score(a) = log(n_a) + γ · z(검증기)_a       (γ 는 val 에서 튜닝한 값)

val 에서 γ=0.5 · ORM 평균이 0.8047 → 0.8200 (구제 5 · 파괴 0) 이었다.
파괴가 0 이라는 게 채택 근거다 —— log(n_a) 항이 다수결을 앵커로 잡아,
검증기가 틀려도 표가 많은 답을 못 뒤집는다.

  python3 src/submit_fuse.py \
      --gen out/gen_qwen3b__lb_C_chat_k64_t0.8_mt3072.jsonl \
      --orm out/orm__orm_r1-merged__<같은이름>.jsonl \
      --ref data/deep_chal_math_leaderboard_filtered.csv \
      --gamma 0.5 --out out/submission.csv
"""
import argparse, collections, json, math, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer, extract_or_fallback


def zs(v):
    ok = [x for x in v if x is not None]
    if len(ok) < 2:
        return [0.0] * len(v)
    mu = sum(ok) / len(ok)
    sd = (sum((x - mu) ** 2 for x in ok) / (len(ok) - 1)) ** .5 or 1.0
    return [0.0 if x is None else (x - mu) / sd for x in v]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--orm", default=None, help="없으면 순수 다수결 (γ 무시)")
    ap.add_argument("--ref", default="data/deep_chal_math_leaderboard_filtered.csv")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--how", default="mean", choices=["mean", "max"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref = pd.read_csv(a.ref)
    ids = list(ref["id"])

    orm = {}
    if a.orm:
        if not Path(a.orm).exists():
            sys.exit(f"검증기 점수 파일이 없다: {a.orm}")
        for line in open(a.orm, encoding="utf-8"):
            try:
                r = json.loads(line); orm[r["id"]] = r["orm"]
            except Exception:
                pass

    pool = {}
    for line in open(a.gen, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        pool[r["id"]] = r["outputs"]

    miss = [i for i in ids if i not in pool]
    if miss:
        sys.exit(f"생성 결과가 없는 문항 {len(miss)}개 (예: {miss[:3]}) —— 생성부터 끝내라")
    if a.orm:
        nocov = [i for i in ids if i not in orm]
        if nocov:
            sys.exit(f"검증기 점수가 없는 문항 {len(nocov)}개 (예: {nocov[:3]})")

    ans, n_fb, n_flip, n_una = [], 0, 0, 0
    for i in ids:
        texts = pool[i]
        preds = [extract_answer(t) for t in texts]
        cnt = collections.Counter(p for p in preds if p is not None)
        if not cnt:
            n_fb += 1
            ans.append(extract_or_fallback(texts[0] if texts else ""))
            continue
        maj = cnt.most_common(1)[0][0]
        if len(cnt) == 1:
            n_una += 1
        if not a.orm:
            ans.append(int(maj)); continue

        z = zs(orm.get(i, [None] * len(texts)))
        agg = collections.defaultdict(list)
        for p, zz in zip(preds, z):
            if p is not None:
                agg[p].append(zz)
        f = (lambda xs: sum(xs) / len(xs)) if a.how == "mean" else max
        best = max(agg.items(), key=lambda kv: math.log(len(kv[1])) + a.gamma * f(kv[1]))[0]
        if best != maj:
            n_flip += 1
        ans.append(int(best))

    sub = pd.DataFrame({"id": ids, "answer": ans})
    sub["answer"] = sub["answer"].astype(int)
    sub.to_csv(a.out, index=False)

    print(f"저장: {a.out}  ({len(sub)}행)")
    print(f"  폴백 {n_fb}개 ({n_fb/len(sub):.2%}) · 만장일치 {n_una}개 ({n_una/len(sub):.1%})")
    if a.orm:
        print(f"  \033[1;36m검증기가 다수결을 뒤집은 문항: {n_flip}개 "
              f"({n_flip/len(sub):.1%})\033[0m")
        if n_flip == 0:
            print("  \033[1;31m!! 뒤집기가 0개다 —— 검증기 점수가 안 붙었거나 γ가 너무 작다\033[0m")
        elif n_flip > len(sub) * 0.25:
            print("  \033[1;31m!! 뒤집기가 25% 초과 —— val(약 6%)과 크게 다르다. γ 확인\033[0m")
    else:
        print("  (순수 다수결 — 검증기 미적용)")


if __name__ == "__main__":
    main()
