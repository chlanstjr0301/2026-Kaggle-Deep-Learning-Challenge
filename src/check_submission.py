"""제출 전 검증. 하루 5회뿐인 제출 기회를 형식 오류로 태우지 않는다.

  python3 src/check_submission.py out/submission_d9.csv \
      --ref data/deep_chal_math_leaderboard_filtered.csv
"""
import argparse, sys
import pandas as pd

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("sub")
    ap.add_argument("--ref", default="data/deep_chal_math_leaderboard_filtered.csv")
    ap.add_argument("--fix", default=None, help="고친 파일을 이 경로로 저장")
    a = ap.parse_args()

    err, warn = [], []
    df = pd.read_csv(a.sub)
    ref = pd.read_csv(a.ref)

    # 1) 컬럼: 정확히 id, answer
    #    Overview 문서에는 "ID"(대문자)로 적혀 있으나 Kaggle 채점기는 소문자 "id"를
    #    요구한다. 문서가 아니라 채점기가 기준이다.
    #    ("ID column id not found in submission" — 2026-08-23 실제 제출에서 확인)
    if list(df.columns) != ["id", "answer"]:
        extra = ("   <- 대문자 ID다. sed -i '1s/^ID,answer$/id,answer/' 로 고칠 것"
                 if list(df.columns) == ["ID", "answer"] else "")
        err.append(f"컬럼이 ['id','answer']가 아님: {list(df.columns)}{extra}")

    # 2) 행 수
    if len(df) != len(ref):
        err.append(f"행 수 {len(df)} != 평가셋 {len(ref)}")

    # 3) id 집합 일치 (정제 버전 id 기준이어야 함)
    idcol = "id" if "id" in df.columns else ("ID" if "ID" in df.columns else None)
    if idcol:
        s, r = set(df[idcol]), set(ref["id"])
        if s != r:
            err.append(f"id 불일치: 누락 {len(r-s)}개, 초과 {len(s-r)}개")
            for x in list(r - s)[:3]: err.append(f"    누락 예: {x}")
            for x in list(s - r)[:3]: err.append(f"    초과 예: {x}")
        if df[idcol].duplicated().any():
            err.append(f"중복 id {df[idcol].duplicated().sum()}개")

    # 4) answer: 결측 없음 + 정수만
    if "answer" in df.columns:
        v = df["answer"]
        if v.isna().any():
            err.append(f"결측 {v.isna().sum()}개")
        nonint = v.dropna().apply(lambda x: float(x) != int(float(x))
                                  if str(x).replace('-','').replace('.','').isdigit() else True)
        if nonint.any():
            err.append(f"정수가 아닌 값 {nonint.sum()}개: "
                       f"{v[nonint].head(3).tolist()}")
        z = (v == 0).sum()
        if z > len(v) * 0.03:
            warn.append(f"answer가 0인 행 {z}개 ({z/len(v):.1%}) — 폴백이 많다는 신호")
        warn.append(f"범위 [{v.min():,}, {v.max():,}]")

    # 5) 학습셋 정답 분포와 대조 — 출력 붕괴나 이상값 쏠림을 잡는다
    try:
        import numpy as np
        tr = pd.read_csv("out/train_clean.csv")["answer"].astype("int64")
        v = df["answer"].dropna().astype("int64")
        rng = np.random.default_rng(0)
        exp = [len(set(rng.choice(tr, len(v), replace=False))) for _ in range(50)]
        warn.append(f"서로 다른 답 {v.nunique()}개 (학습셋 기준 기대 {min(exp)}~{max(exp)})")
        if v.nunique() < min(exp) * 0.7:
            err.append("서로 다른 답이 지나치게 적다 — 출력이 붕괴했을 수 있다")
        warn.append(f"중앙값 {v.median():.0f} (학습셋 {tr.median():.0f}) · "
                    f"|x|>1e6 {(v.abs()>1e6).mean():.2%} (학습셋 {(tr.abs()>1e6).mean():.2%}) · "
                    f"음수 {(v<0).mean():.2%} (학습셋 {(tr<0).mean():.2%})")
    except Exception:
        pass

    print(f"파일: {a.sub}   {len(df)}행")
    for e in err:  print(f"  \033[1;31m[오류]\033[0m {e}")
    for w in warn: print(f"  \033[1;33m[참고]\033[0m {w}")

    if a.fix and not err:
        df = df.set_index(idcol).reindex(ref["id"]).reset_index()
        df.columns = ["id", "answer"]
        df["answer"] = df["answer"].fillna(0).astype(int)
        df.to_csv(a.fix, index=False)
        print(f"  정렬본 저장: {a.fix}")

    print("\n" + ("\033[1;32m제출 가능\033[0m" if not err
                  else "\033[1;31m제출하지 말 것 — 위 오류 먼저 해결\033[0m"))
    sys.exit(1 if err else 0)

if __name__ == "__main__":
    main()
