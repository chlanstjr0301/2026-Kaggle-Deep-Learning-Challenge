"""공개 코퍼스에서 우리 과제에 맞는 부분만 골라 SFT 형식으로 만든다.

규정 5.2a: 공개·무료·동등 접근 가능한 데이터만. 게이트(연락처 동의) 저장소는 쓰지 않는다.
규정 5.2c: 사용 데이터셋 목록을 최종 제출 시 명시해야 하므로 출처를 함께 기록한다.

선별 규칙 — 우리 채점 형식이 정해준다:
  1) 최종 답이 **정수**인 것만                (exact-match 정수 채점)
  2) 풀이 길이 상한                           (3072 토큰 예산과 정합)
  3) 리더보드/val 문항과 겹치는 것 제거        ← 모르는 채로 평가셋을 학습하는 사고 방지
  4) 문제 길이 하한/상한                      (우리 분포: 중앙 200자)

  python3 src/build_external.py --sets orca,omi2 --max-per-set 60000
"""
import argparse, json, re, sys, unicodedata, collections
from pathlib import Path
import pandas as pd

SETS = {
    # key: (repo, config, split, license, question_field, solution_field)
    "orca":  ("microsoft/orca-math-word-problems-200k", None, "train",
              "MIT", "question", "answer"),
    "omi2":  ("nvidia/OpenMathInstruct-2", None, "train_1M",
              "CC-BY-4.0", "problem", "generated_solution"),
    "numina":("AI-MO/NuminaMath-1.5", None, "train",
              "Apache-2.0", "problem", "solution"),
}

_BOX = re.compile(r"\\boxed\s*\{")
_INT = re.compile(r"^-?\d{1,17}$")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[^a-z0-9]", "", s)[:200]


def boxed(text: str):
    m = _BOX.search(text)
    if not m:
        return None
    i, d = m.end(), 1
    for j in range(i, len(text)):
        if text[j] == "{": d += 1
        elif text[j] == "}":
            d -= 1
            if d == 0:
                return text[i:j]
    return None


def final_int(sol: str):
    """풀이의 최종 답이 정수면 그 값, 아니면 None."""
    b = boxed(sol)
    if b is not None:
        b = b.replace(",", "").replace("$", "").strip()
        return int(b) if _INT.match(b) else None
    tail = sol.strip().splitlines()[-1] if sol.strip() else ""
    nums = re.findall(r"-?\d[\d,]*", tail)
    if len(nums) != 1:
        return None
    v = nums[0].replace(",", "")
    return int(v) if _INT.match(v) else None


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--sets", default="orca,omi2")
    ap.add_argument("--max-per-set", type=int, default=60000)
    ap.add_argument("--max-sol-chars", type=int, default=2400)
    ap.add_argument("--min-sol-chars", type=int, default=120)
    ap.add_argument("--min-q-chars", type=int, default=40)
    ap.add_argument("--max-q-chars", type=int, default=1200)
    ap.add_argument("--holdout", default="data/deep_chal_math_leaderboard_filtered.csv,out/val.csv",
                    help="오염 제거 대상 (쉼표 구분)")
    ap.add_argument("--out", default="out/external_sft.jsonl")
    ap.add_argument("--manifest", default="out/external_manifest.json")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--require-boxed", action="store_true",
                    help="원본에 \\boxed{} 가 있는 것만 채택 (가장 보수적)")
    a = ap.parse_args()

    from datasets import load_dataset

    ban = set()
    for p in a.holdout.split(","):
        p = p.strip()
        if p and Path(p).exists():
            d = pd.read_csv(p)
            col = "question" if "question" in d.columns else d.columns[1]
            ban |= {norm(x) for x in d[col]}
    print(f"[오염제거] 보류 문항 {len(ban):,}개 지문 등록")

    rows, manifest = [], []
    for key in [s.strip() for s in a.sets.split(",") if s.strip()]:
        if key not in SETS:
            print(f"  !! 알 수 없는 셋: {key} (가능: {list(SETS)})"); continue
        repo, cfg, split, lic, qf, sf = SETS[key]
        print(f"\n[{key}] {repo} · {split} · {lic}")
        ds = load_dataset(repo, cfg, split=split, streaming=a.streaming)
        st = collections.Counter()
        kept = 0
        for r in ds:
            st["본"] += 1
            q, s = str(r.get(qf, "")), str(r.get(sf, ""))
            if not q or not s:
                st["빈 필드"] += 1; continue
            if not (a.min_q_chars <= len(q) <= a.max_q_chars):
                st["문제 길이"] += 1; continue
            if not (a.min_sol_chars <= len(s) <= a.max_sol_chars):
                st["풀이 길이"] += 1; continue
            v = final_int(s)
            if v is None:
                st["정수 아님"] += 1; continue
            if norm(q) in ban:
                st["오염 제거"] += 1; continue
            had_box = boxed(s) is not None
            if a.require_boxed and not had_box:
                st["boxed 없음"] += 1; continue
            # ---- 출력 형식을 우리 추론 형식에 맞춘다 -------------------------
            # 지금 우리 파싱실패율은 0.00% 다. \boxed{} 없는 풀이로 학습하면
            # 모델이 \boxed{} 를 그만 쓰게 되고, 그 0.00% 가 통째로 돌아온다.
            # 추론 프롬프트(C)가 요구하는 형식으로 꼬리를 통일한다.
            if not had_box:
                s = s.rstrip() + f"\n\nThe final answer is \\boxed{{{v}}}."
                st["boxed 보강"] += 1
            else:
                st["boxed 원본"] += 1
            rows.append({"id": f"{key}-{kept:07d}", "question": q,
                         "solution": s, "answer": v, "label": 1,
                         "src": key, "had_box": had_box})
            kept += 1
            st["채택"] += 1
            if kept >= a.max_per_set:
                break
        for k, v in st.most_common():
            print(f"    {k:<12} {v:,}")
        manifest.append({"key": key, "repo": repo, "config": cfg, "split": split,
                         "license": lic, "kept": kept, "stats": dict(st)})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    by = collections.Counter(r["src"] for r in rows)
    ln = sorted(len(r["solution"]) for r in rows)
    print(f"\n총 {len(rows):,}건  " + " · ".join(f"{k} {v:,}" for k, v in by.items()))
    if ln:
        print(f"풀이 길이 중앙값 {ln[len(ln)//2]}자 · 95분위 {ln[int(len(ln)*.95)]}자")
    nb = sum(1 for r in rows if r.get("had_box"))
    print(f"원본에 \\boxed{{}} 있던 것 {nb:,} ({nb/max(len(rows),1):.1%}) · "
          f"보강한 것 {len(rows)-nb:,}  ← 전부 \\boxed{{}} 로 끝나게 통일됨")
    bysrc = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        bysrc[r["src"]][0] += 1; bysrc[r["src"]][1] += bool(r.get("had_box"))
    for k, (n, b) in bysrc.items():
        print(f"    {k}: {n:,}건 중 원본 boxed {b:,} ({b/max(n,1):.1%})")
    print(f"저장: {a.out}\n출처 기록: {a.manifest}  ← 최종 제출 시 이 목록을 명시할 것")


if __name__ == "__main__":
    main()
