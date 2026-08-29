"""데이터 정제 + val 층화 분리.

제거 대상 4종의 합집합:
  1) train_filtered_ids.csv        (주최 측 공식, 627)
  2) mislabel_442.csv              (참가자 제보, 디스코드)
  3) illposed_623.csv              (참가자 제보, 디스코드)
  4) 정규식 패턴                    (이미지 참조, asy, 역질문 등)

2~3은 없으면 건너뛰고 경고만 낸다. 나중에 받으면 재실행할 것.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42

# 텍스트만으로 풀 수 없거나 라벨 신뢰도가 낮은 패턴
DROP_PATTERNS = {
    "image_ref":   r"!\[|cdn\.mathpix|artofproblemsolving\.com/wiki/index\.php/File:|\.png|\.jpg",
    "asy_code":    r"\[asy\]",
    "url":         r"https?://",
    "reverse_q":   r"what is the value of unknown variable",
}

# 운영진 검수 기준 2번("불필요한 지시문이나 소실된 문구")에 해당하지만,
# 문제 본문 자체는 멀쩡한 경우가 많다. 제거하지 말고 꼬리만 잘라낸다.
# 최종 test는 "완벽히 정비된 데이터셋"이라 했으므로, 이 정리가 오히려 형식을 맞춘다.
SANITIZE_PATTERNS = [
    r"\s*Translate the (?:above )?text (?:above )?into English[^\n]*",
    r"\s*Translate the above (?:text|content)[^\n]*",
    r"\s*please (?:retain|keep) the (?:original )?[^\n]*line breaks[^\n]*",
    r"\s*(?:and )?output the translation result(?: directly)?\.?",
    r"\s*directly output the translation result\.?",
]

def sanitize(q: str) -> str:
    for p in SANITIZE_PATTERNS:
        q = re.sub(p, "", q, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", q).strip()


def _read_ids(data_dir: Path, *names):
    for n in names:
        p = data_dir / n
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception as e:
                print(f"  [warn] {n} 읽기 실패: {e}", file=sys.stderr)
    return None


def load_bad_ids(data_dir: Path) -> tuple[set, set, dict, pd.DataFrame | None]:
    """(제거 id, 격리 id, 출처별 집계, 격리 상세)를 반환.

    mislabel 목록은 제안 정답이 함께 오므로 즉시 버리지 않고 격리한다.
    운영진도 리더보드 검수에서 "정답이 명확히 확인되는 문항은 정답을 수정하여
    유지"했으므로, 재라벨링은 승인된 처리 방식이다. 다만 제안값은 다른
    참가자의 계산이므로 D-7의 만장일치 검사로 독립 확인한 뒤에 편입한다.
    """
    bad, prov = set(), {}
    for name, *files in [
        ("official_627", "train_filtered_ids.csv"),
        ("illposed",     "illposed_623.csv", "organizer_report_illposed_623.csv"),
    ]:
        df = _read_ids(data_dir, *files)
        if df is None:
            continue
        ids = set(df["id"])
        prov[name] = {"total": len(ids), "new": len(ids - bad)}
        bad |= ids

    quar_df = _read_ids(data_dir, "mislabel_442.csv", "organizer_report_mislabel_442.csv")
    quar = set()
    if quar_df is not None:
        quar = set(quar_df["id"]) - bad          # 이미 확정 제거된 것은 격리하지 않는다
        prov["mislabel(격리)"] = {"total": len(quar_df), "new": len(quar)}
        quar_df = quar_df[quar_df["id"].isin(quar)].reset_index(drop=True)
    return bad, quar, prov, quar_df

def clean(df: pd.DataFrame, bad: set, quar: set) -> tuple[pd.DataFrame, dict]:
    stats = {"input": len(df)}
    mask_id = df["id"].isin(bad | quar)
    stats["by_id_list"] = int(df["id"].isin(bad).sum())
    stats["quarantined"] = int(df["id"].isin(quar).sum())

    mask_pat = pd.Series(False, index=df.index)
    for name, pat in DROP_PATTERNS.items():
        m = df["question"].str.contains(pat, case=False, regex=True, na=False)
        stats[f"pat_{name}"] = int((m & ~mask_id & ~mask_pat).sum())  # 신규 제거분만
        mask_pat |= m

    keep = ~(mask_id | mask_pat)
    out = df[keep].reset_index(drop=True)

    # 제거가 아니라 정리: 번역 지시문 꼬리만 잘라낸다
    before = out["question"].copy()
    out["question"] = out["question"].map(sanitize)
    stats["sanitized"] = int((before != out["question"]).sum())
    stats["removed_total"] = int((~keep).sum())
    stats["output"] = len(out)
    return out, stats

def stratify_split(df: pd.DataFrame, n_val: int, seed: int = SEED):
    """문항 길이 4분위 x LaTeX 유무 = 8층에서 비례 추출."""
    d = df.copy()
    d["_len_q"] = pd.qcut(d["question"].str.len(), 4, labels=False, duplicates="drop")
    d["_tex"] = d["question"].str.contains(r"\$|\\frac|\\text|\\sqrt", regex=True, na=False).astype(int)
    d["_stratum"] = d["_len_q"].astype(str) + "_" + d["_tex"].astype(str)

    rng = np.random.RandomState(seed)
    picks = []
    for s, g in d.groupby("_stratum"):
        k = int(round(n_val * len(g) / len(d)))
        k = min(k, len(g))
        if k > 0:
            picks.append(g.sample(n=k, random_state=rng))
    val = pd.concat(picks).sort_index()

    # 반올림 오차 보정
    diff = n_val - len(val)
    if diff > 0:
        pool = d.drop(val.index)
        val = pd.concat([val, pool.sample(n=diff, random_state=rng)]).sort_index()
    elif diff < 0:
        val = val.sample(n=n_val, random_state=rng).sort_index()

    train = d.drop(val.index).reset_index(drop=True)
    cols = ["id", "question", "answer"]
    return train[cols], val[cols].reset_index(drop=True), d.loc[val.index, "_stratum"].value_counts().to_dict()

def profile(df: pd.DataFrame, name: str) -> dict:
    q = df["question"]
    return {
        "name": name, "n": len(df),
        "len_median": float(q.str.len().median()),
        "len_mean": round(float(q.str.len().mean()), 1),
        "tex_ratio": round(float(q.str.contains(r"\$|\\frac|\\text", regex=True, na=False).mean()), 4),
    }

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    dd, od = Path(a.data_dir), Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(dd / "deep_chal_math_train.csv")
    bad, quar, prov, quar_df = load_bad_ids(dd)

    print(f"== 제거 목록 ==")
    for k, v in prov.items():
        print(f"  {k:14s} 총 {v['total']:5d}  신규 {v['new']:5d}")
    missing = [f for f in ["mislabel_442.csv", "illposed_623.csv"] if not (dd / f).exists()
               and not (dd / f"organizer_report_{f}").exists()]
    if missing:
        print(f"  [!] 미발견: {missing}  → 디스코드에서 받은 뒤 재실행할 것")
    print(f"  확정 제거 {len(bad)}건 · 격리 {len(quar)}건 · 합 {len(bad|quar)}건\n")

    clean_df, stats = clean(train_raw, bad, quar)
    print("== 정제 결과 ==")
    for k, v in stats.items():
        print(f"  {k:18s} {v}")
    print()

    train, val, strata = stratify_split(clean_df, a.n_val, a.seed)

    lb_path = dd / "deep_chal_math_leaderboard_filtered.csv"
    profs = [profile(train, "train"), profile(val, "val")]
    if lb_path.exists():
        profs.append(profile(pd.read_csv(lb_path), "leaderboard"))

    print("== 분포 대조 (val이 leaderboard를 닮아야 한다) ==")
    print(f"  {'집합':12s} {'N':>6s} {'길이중앙':>8s} {'길이평균':>8s} {'LaTeX':>7s}")
    for p in profs:
        print(f"  {p['name']:12s} {p['n']:6d} {p['len_median']:8.0f} {p['len_mean']:8.1f} {p['tex_ratio']:7.3f}")

    train.to_csv(od / "train_clean.csv", index=False)
    val.to_csv(od / "val.csv", index=False)
    if quar_df is not None and len(quar_df):
        keep = [c for c in ["id", "question", "current_label", "suggested_answer"] if c in quar_df.columns]
        quar_df[keep].to_csv(od / "train_quarantine.csv", index=False)
        print(f"저장: {od/'train_quarantine.csv'} ({len(quar_df)}) "
              f"— D-7 만장일치 검사로 편입/폐기 판정")

    meta = {"seed": a.seed, "provenance": prov, "clean_stats": stats,
            "profiles": profs, "val_strata": strata, "missing_lists": missing,
            "quarantine_n": len(quar)}
    (od / "clean_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n저장: {od/'train_clean.csv'} ({len(train)}), {od/'val.csv'} ({len(val)}), {od/'clean_meta.json'}")

if __name__ == "__main__":
    main()
