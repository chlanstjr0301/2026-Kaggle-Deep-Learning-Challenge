"""다수결 + 검증기 융합. 게인 γ 를 val 에서 튜닝한다.

    score(a) = log(n_a) + γ · zscore(검증기 점수)_a

γ=0 이면 순수 다수결, γ→∞ 면 순수 Best-of-N. 시뮬레이션은 그 사이에
최적이 있다고 말했다 (칼만 게인의 정적 특수경우). 여기서 실제로 확인한다.

같이 출력하는 것:
  - 문항 내 pairwise AUC — 검증기의 진짜 품질. pooled AUC 는 "어려운 문제 = 오답"을
    배운 검증기도 높게 나오므로 쓰면 안 된다.
  - 로그확률(logp)만 쓴 융합 — 학습 없는 대조군

  python3 src/fuse.py --gen out/gen_qwen3b__val_C_chat_k32_t0.8_mt3072.jsonl \
      --orm out/orm__orm_r1-merged__gen_qwen3b__val_C_chat_k32_t0.8_mt3072.jsonl --m 32
"""
import argparse, collections, json, math, random, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer

GAMMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 1e9]


def zs(v):
    ok = [x for x in v if x is not None]
    if len(ok) < 2:
        return [0.0] * len(v)
    mu = sum(ok) / len(ok)
    sd = (sum((x - mu) ** 2 for x in ok) / (len(ok) - 1)) ** .5 or 1.0
    return [0.0 if x is None else (x - mu) / sd for x in v]


def pick(items, g, how):
    """items: [(answer, z)] -> 최종 답.

    동률은 **답의 크기 오름차순**으로 깬다. 임의의 규칙이지만 결정적이어야 한다 ——
    표본 순서에 따라 갈리게 두면 같은 데이터에서 두 코드가 다른 답을 낸다
    (실제로 fuse.py 안에서 다수결이 두 번 구현돼 있었고 300문항 중 4개가 어긋났다).
    """
    agg = collections.defaultdict(list)
    for a, z in items:
        if a is not None:
            agg[a].append(z)
    if not agg:
        return None
    if g >= 1e8:                                   # 순수 Best-of-N
        return max(sorted(agg.items()), key=lambda kv: max(kv[1]))[0]
    f = (lambda xs: sum(xs) / len(xs)) if how == "mean" else max
    return max(sorted(agg.items()),
               key=lambda kv: math.log(len(kv[1])) + g * f(kv[1]))[0]


def within_auc(pool, gold, key, only=None):
    """문항 내 pairwise AUC. 같은 문항의 (정답, 오답) 쌍만 센다.
    only: 문항 id 집합으로 제한 (None이면 전체)."""
    win = tie = tot = 0
    for i, rows in pool.items():
        if only is not None and i not in only:
            continue
        pos = [r[key] for r in rows if r["ans"] == gold[i] and r[key] is not None]
        neg = [r[key] for r in rows if r["ans"] != gold[i] and r[key] is not None]
        for p in pos:
            for n in neg:
                tot += 1
                if p > n: win += 1
                elif p == n: tie += 1
    return (win + 0.5 * tie) / tot if tot else float("nan"), tot


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--orm", default=None)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--out", default="out/fuse_report.json")
    a = ap.parse_args()

    df = pd.read_csv(a.val).head(a.limit) if a.limit else pd.read_csv(a.val)
    gold = dict(zip(df["id"], df["answer"]))

    orm = {}
    if a.orm and Path(a.orm).exists():
        with open(a.orm, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line); orm[r["id"]] = r["orm"]
                except Exception:
                    pass

    pool = {}
    with open(a.gen, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            i = r["id"]
            if i not in gold:
                continue
            ts = r["outputs"]
            lp = r.get("logp", [None] * len(ts))
            ov = orm.get(i, [None] * len(ts))
            zo, zl = zs(ov), zs(lp)
            pool[i] = [{"ans": extract_answer(t), "orm": (ov[j] if j < len(ov) else None),
                        "zorm": zo[j], "zlp": zl[j],
                        "logp": (lp[j] if j < len(lp) else None)}
                       for j, t in enumerate(ts)]
    ids = [i for i in df["id"] if i in pool]
    m = min(a.m, min(len(v) for v in pool.values()))
    print(f"문항 {len(ids)} · m={m} · {a.trials}회 · 검증기 {'있음' if orm else '없음'}\n")

    # logp 결측을 조용히 넘기지 않는다 —— 전부 null이면 AUC 0.5, 융합 Δ 0 이 나오는데
    # 그건 "신호가 없다"가 아니라 "생성 때 --logprobs 를 안 켰다"는 뜻이다 (E15).
    _tot = sum(len(v) for v in pool.values())
    _have = sum(1 for v in pool.values() for x in v if x.get("logp") is not None)
    if _have == 0:
        print("\033[1;31m[경고] logp 가 하나도 없다 — 'logp·평균' 대조군은 무의미하다."
              " 생성 때 --logprobs 필요.\033[0m\n")

    for key, name in (("orm", "검증기"), ("logp", "로그확률")):
        if key == "orm" and not orm:
            continue
        if key == "logp" and _have == 0:
            continue
        auc, n = within_auc(pool, gold, key)
        print(f"[{name}] 문항 내 pairwise AUC = {auc:.4f}  ({n:,}쌍)")
    print()

    # ---- 어디서 실패하는가 -------------------------------------------------
    # 전체 AUC 는 다수결이 이미 맞히는 쉬운 문항에서 벌어들인 값이 대부분이다.
    # 정작 검증기가 값을 해야 할 곳은 '다수결이 틀린 문항'이다. 거기서도 AUC 가
    # 높아야 진짜 실력이고, 0.5 근처면 "쉬운 문제에서만 잘하는" 검증기다.
    # 같은 베이스 모델에서 나왔으므로 오류가 상관될 수 있다 —— 그게 이 분리의 이유다.
    def _maj(i):
        # pick(..., g=0) 과 반드시 같은 규칙을 써야 한다
        return pick([(r["ans"], 0.0) for r in pool[i]], 0.0, "mean")
    MW = {i for i in ids if _maj(i) != gold[i]}
    MR = {i for i in ids if _maj(i) == gold[i]}
    print(f"{'':>10} {'다수결오답':>10} {'다수결정답':>10}   (문항 {len(MW)} / {len(MR)})")
    for key, name in (("orm", "검증기"), ("logp", "로그확률")):
        if key == "orm" and not orm:      continue
        if key == "logp" and _have == 0:  continue
        aw, nw = within_auc(pool, gold, key, MW)
        ar, nr = within_auc(pool, gold, key, MR)
        print(f"{name:>10} {aw:>10.4f} {ar:>10.4f}")
    print("  ↑ 왼쪽 열이 0.5 근처면 검증기가 정답성이 아니라 문제 난이도를 배운 것이다.\n")

    rng0 = random.Random(42)
    subs = {i: [rng0.sample(range(len(pool[i])), m) for _ in range(a.trials)] for i in ids}

    def run(zkey, g, how):
        hit = 0.0
        for i in ids:
            ok = 0
            for idx in subs[i]:
                items = [(pool[i][j]["ans"], pool[i][j][zkey]) for j in idx]
                ok += (pick(items, g, how) == gold[i])
            hit += ok / a.trials
        return hit / len(ids)

    base = run("zorm", 0.0, "mean")     # γ=0 이면 zkey 무관
    print(f"{'γ':>6}  " + "  ".join(f"{s:>9}" for s in
          (["ORM·평균", "ORM·최대"] if orm else []) + ["logp·평균"]))
    print("-" * (8 + 11 * (3 if orm else 1)))
    res = {}
    for g in GAMMAS:
        cells, row = [], {}
        for zkey, how, tag in ([("zorm", "mean", "orm_mean"), ("zorm", "max", "orm_max")]
                               if orm else []) + [("zlp", "mean", "logp_mean")]:
            v = run(zkey, g, how)
            row[tag] = round(v, 4); cells.append(f"{v:>9.4f}")
        res["BoN" if g >= 1e8 else f"{g:g}"] = row
        print(f"{('BoN' if g>=1e8 else f'{g:g}'):>6}  " + "  ".join(cells))
    print("-" * (8 + 11 * (3 if orm else 1)))
    print(f"γ=0 (순수 다수결) = {base:.4f}")

    best = max(((g, t, v) for g, r in res.items() for t, v in r.items()),
               key=lambda x: x[2])
    print(f"\n최적: γ={best[0]} · {best[1]} · {best[2]:.4f}  ({best[2]-base:+.4f})")

    # ---- 검증기가 실제로 값을 하는 지점만 따로 본다 -------------------------
    # 전체 정확도는 "다수결이 이미 맞히는 79%"에 희석돼 좋은 검증기도 밋밋해 보인다.
    # 진짜 질문 두 개: (a) 다수결이 틀린 문항을 몇 개 구했나 (b) 맞던 걸 몇 개 깼나
    _gmap = {"BoN": 1e9}
    bg = _gmap.get(best[0], None)
    if bg is None:
        bg = float(best[0])
    bzkey, bhow = (("zlp", "mean") if best[1] == "logp_mean"
                   else ("zorm", "max" if best[1] == "orm_max" else "mean"))
    save = broke = mw = mr = 0
    for i in ids:
        for idx in subs[i]:
            it0 = [(pool[i][j]["ans"], 0.0) for j in idx]
            it1 = [(pool[i][j]["ans"], pool[i][j][bzkey]) for j in idx]
            a0 = pick(it0, 0.0, "mean") == gold[i]
            a1 = pick(it1, bg, bhow) == gold[i]
            if a0: mr += 1
            else:  mw += 1
            if not a0 and a1: save += 1
            if a0 and not a1: broke += 1
    print(f"\n[최적 γ에서의 분해]  다수결 오답 {mw/a.trials:.0f}문항 중 "
          f"\033[1;32m구제 {save/a.trials:.1f}\033[0m ({save/max(mw,1):.1%})  ·  "
          f"다수결 정답 {mr/a.trials:.0f}문항 중 "
          f"\033[1;31m파괴 {broke/a.trials:.1f}\033[0m ({broke/max(mr,1):.1%})")
    print(f"                   순이득 {(save-broke)/a.trials/len(ids):+.4f}"
          f"  (구제율이 파괴율보다 커야 검증기가 값을 한다)")
    n = len(ids)
    print(f"참고: n={n} 절대 SE = {math.sqrt(base*(1-base)/n):.4f} "
          f"(규칙 간 비교는 동일 부분표본이라 더 민감)")

    Path(a.out).write_text(json.dumps(
        {"n": n, "m": m, "baseline": round(base, 4), "grid": res,
         "best": {"gamma": best[0], "how": best[1], "acc": best[2]}},
        ensure_ascii=False, indent=2))
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
