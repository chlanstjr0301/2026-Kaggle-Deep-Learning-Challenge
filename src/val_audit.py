"""val 라벨 감사 —— 우리 모델이 틀린 문항을 교사에게 독립적으로 풀린다.

목적은 점수를 올리는 게 아니라 **측정기를 검증하는 것**이다.

세 갈래로 갈린다:
  교사 == gold                  -> 우리 모델이 못 푸는 진짜 어려운 문제
  교사 == 우리 다수결 != gold    -> **두 독립 시스템이 라벨에 맞선다. 오라벨 강한 증거**
  교사 != gold, != 우리 다수결   -> 판정 보류

이것이 왜 중요한가: pass@32 = 0.92 라는 우리 천장에 오라벨이 섞여 있으면
"구제 불가 24문항"의 일부는 애초에 답이 없는 문제다. 그러면 지금까지의 모든
γ 튜닝과 판정선이 조금씩 어긋나 있었다는 뜻이 된다.

  export DEEPSEEK_API_KEY=...
  python3 src/val_audit.py --gen out/gen_qwen3b__val_C_chat_k32_t0.8_mt3072_lp.jsonl \
      --limit 300 --k 4 --out out/val_audit.jsonl
"""
import argparse, collections, json, os, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer
from teach_gen import call, PRICE, _stat


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--val", default="out/val.csv")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scope", default="maj_wrong", choices=["maj_wrong", "dead"],
                    help="maj_wrong=다수결이 틀린 문항 / dead=정답이 0개인 문항")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY 없음")

    d = pd.read_csv(a.val).head(a.limit) if a.limit else pd.read_csv(a.val)
    gold = dict(zip(d["id"], d["answer"]))
    q = dict(zip(d["id"], d["question"]))

    pool = {}
    for line in open(a.gen, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in gold:
            pool[r["id"]] = [extract_answer(t) for t in r["outputs"]]

    tgt = []
    for i, preds in pool.items():
        c = collections.Counter(x for x in preds if x is not None)
        maj = c.most_common(1)[0][0] if c else None
        dead = not any(x == gold[i] for x in preds)
        if (a.scope == "dead" and dead) or (a.scope == "maj_wrong" and maj != gold[i]):
            tgt.append((i, maj, dead))

    done = set()
    if Path(a.out).exists():
        for line in open(a.out, encoding="utf-8"):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    todo = [t for t in tgt if t[0] not in done]
    print(f"[audit] 대상 {len(tgt)} ({a.scope}) · 완료 {len(done)} · 남음 {len(todo)} "
          f"· 문항당 {a.k}회", flush=True)
    if not todo:
        print("남은 작업 없음")
    else:
        f = open(a.out, "a", encoding="utf-8")
        lock = threading.Lock()
        def work(t):
            i, maj, dead = t
            outs = [call(a.model, str(q[i]), a.temp, key) for _ in range(a.k)]
            outs = [o for o in outs if o]
            with lock:
                f.write(json.dumps({"id": i, "gold": int(gold[i]),
                                    "our_maj": None if maj is None else int(maj),
                                    "dead": dead, "outputs": outs},
                                   ensure_ascii=False) + "\n")
                f.flush(); os.fsync(f.fileno())
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, todo))
        f.close()
        pi, po = PRICE.get(a.model, (0.435, 0.87))
        print(f"실비 ${_stat['in']/1e6*pi + _stat['out']/1e6*po:.2f}")

    # ── 판정 ──────────────────────────────────────────────────────────
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    cat = collections.Counter(); mis = []
    for r in rows:
        ta = [extract_answer(t) for t in r["outputs"]]
        ta = [x for x in ta if x is not None]
        if not ta:
            cat["교사 파싱실패"] += 1; continue
        c = collections.Counter(ta); tmaj, n = c.most_common(1)[0]
        unan = (n == len(ta) and len(ta) >= 3)
        if tmaj == r["gold"]:
            cat["교사=gold (진짜 어려움)"] += 1
        elif r["our_maj"] is not None and tmaj == r["our_maj"]:
            cat["교사=우리다수결≠gold (오라벨 의심)"] += 1
            mis.append((r["id"], r["gold"], tmaj, n, len(ta), r["dead"]))
        else:
            cat["삼자 불일치 (보류)"] += 1
    print(f"\n감사 문항 {len(rows)}")
    for k, v in cat.most_common():
        print(f"  {k:<34} {v:>4}  ({v/max(len(rows),1):>5.1%})")
    if mis:
        print(f"\n오라벨 의심 {len(mis)}건:")
        for i, g, t, n, m, dead in mis[:20]:
            print(f"  {i}  gold={g:<12} 교사={t:<12} ({n}/{m} 합의)"
                  + ("  [정답0]" if dead else ""))
        ids = [x[0] for x in mis]
        Path("out/val_suspect.json").write_text(json.dumps(ids, ensure_ascii=False))
        print(f"\n저장: out/val_suspect.json ({len(ids)}건)")
        print("  → 이 문항들을 제외하고 재측정하면 우리 지표의 참값을 볼 수 있다")


if __name__ == "__main__":
    main()
