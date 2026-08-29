#!/bin/bash
# 최종 제출 파이프라인. 사용법: ./run_final.sh <입력.parquet|csv> [태그]
# 구성: 베이스 Qwen2.5-3B · 프롬프트 C · chat · k=64 · T=0.8 · mt3072 · ORM 융합 γ=0.5
set -uo pipefail
IN="${1:-}"; TAG="${2:-final}"
MODEL="./qwen3b"; ORM="ckpt/orm_r2-merged"; K=64; TEMP=0.8; MT=3072; GAMMA=0.5
# 주의: ORM 은 병합 모델 경로다 (어댑터 ckpt/orm_r1 아님). 없으면 ./merge_orm.sh
T_START=$(date +%s)
say(){ echo -e "\n\033[1;36m═══ $* ═══\033[0m"; }
die(){ echo -e "\033[1;31m✗ $*\033[0m"; exit 1; }
ok(){ echo -e "  \033[32m✓\033[0m $*"; }

# ─────────────────────────── [0] 사전점검 (3초) ───────────────────────────
say "0/6 사전점검"
[ -n "$IN" ] || die "입력 파일을 지정해라: ./run_final.sh test.parquet"
[ -f "$IN" ] || die "입력 파일 없음: $IN"; ok "입력 $IN ($(du -h "$IN"|cut -f1))"
[ -f "$MODEL/config.json" ] || die "모델 없음: $MODEL/config.json"; ok "모델 $MODEL"
[ -f "$ORM/config.json" ] || die "ORM 병합본 없음: $ORM/config.json\n     복구: ./merge_orm.sh  (어댑터 ckpt/orm_r1 + 베이스 ./qwen3b 로 재생성)"
ok "ORM $ORM"
for f in src/baseline.py src/score_orm.py src/submit_fuse.py; do
  [ -f "$f" ] || die "스크립트 없음: $f"; done; ok "스크립트 3종"
python3 - <<'PY' || die "필수 모듈 누락 (위 목록 확인)"
import sys
miss=[]
for m in ("vllm","torch","transformers","pandas","numpy","peft"):
    try: __import__(m)
    except Exception as e: miss.append(f"{m}: {type(e).__name__}")
if miss: print("  누락:", *miss, sep="\n   "); sys.exit(1)
import vllm, torch
print(f"  vllm {vllm.__version__} · torch {torch.__version__}")
PY
ok "모듈"
AV=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${AV:-0}" -ge 20 ] || die "디스크 여유 ${AV}G < 20G"; ok "디스크 ${AV}G"
if command -v nvidia-smi >/dev/null; then
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
  [ "$FREE" -ge 30000 ] || die "GPU 여유 ${FREE}MiB < 30000. 다른 프로세스 확인: nvidia-smi"
  ok "GPU 여유 ${FREE}MiB"
fi
pgrep -f "baseline.py|score_orm.py|train_sft.py" >/dev/null && die "GPU 작업이 이미 돌고 있다"
ok "GPU 점유 없음"

# ─────────────────────────── [1] 입력 정규화 ───────────────────────────
say "1/6 입력 정규화"
CSV="out/${TAG}_input.csv"
python3 - "$IN" "$CSV" <<'PY' || die "입력 변환 실패"
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src)
cols = {c.lower(): c for c in df.columns}
qc = next((cols[k] for k in ("question","problem","text","query") if k in cols), None)
ic = next((cols[k] for k in ("id","idx","index","problem_id") if k in cols), None)
if qc is None: sys.exit(f"question 컬럼을 못 찾음. 있는 컬럼: {list(df.columns)}")
out = pd.DataFrame({"id": df[ic] if ic else [f"t-{i:06d}" for i in range(len(df))],
                    "question": df[qc]})
ac = next((cols[k] for k in ("answer","gold","label","target") if k in cols), None)
if ac is not None and df[ac].notna().any():
    out["answer"] = df[ac]
else:
    ac = None
    out["answer"] = -999999999   # 정답 없는 평가셋. 하류 코드가 컬럼을 요구한다
assert out["question"].notna().all(), "question 에 결측"
assert out["id"].is_unique, "id 중복"
out.to_csv(dst, index=False)
open(dst + ".n", "w").write(str(len(out)))   # wc -l 은 인용문 내 줄바꿈 때문에 못 쓴다
print(f"  {len(out):,}문항 · 정답열 {'있음' if ac else '없음'} → {dst}")
PY
N=$(cat "$CSV.n"); ok "$N 문항"

# ─────────────────────────── [2] 생성 ───────────────────────────
say "2/6 생성 (k=$K · 체크포인트 재개 가능)"
T2=$(date +%s)
python3 src/baseline.py --model "$MODEL" --val "$CSV" --limit "$N" \
  --prompts C --modes chat --k "$K" --temp "$TEMP" --max-tokens "$MT" --logprobs \
  || die "생성 실패. 같은 명령을 다시 실행하면 이어서 진행된다"
GEN=$(ls -t out/gen_*"$(basename "$CSV" .csv)"_C_chat_k${K}_t${TEMP}_mt${MT}_lp.jsonl 2>/dev/null | head -1)
[ -n "$GEN" ] && [ -f "$GEN" ] || die "생성 파일을 못 찾음"
GN=$(wc -l < "$GEN"); [ "$GN" -eq "$N" ] || die "생성 $GN행 ≠ 입력 $N행"
ok "$GEN ($GN행 · $((($(date +%s)-T2)/60))분)"

# ─────────────────────────── [3] ORM 채점 ───────────────────────────
say "3/6 ORM 채점"
T3=$(date +%s); OSC="out/orm__${TAG}.jsonl"
python3 src/score_orm.py --orm "$ORM" --gen "$GEN" --val "$CSV" --out "$OSC" \
  || die "ORM 채점 실패"
[ -s "$OSC" ] || die "ORM 산출물이 비었다"
ok "$OSC ($(wc -l < "$OSC")행 · $((($(date +%s)-T3)/60))분)"

# ─────────────────────────── [4] 제출 파일 ───────────────────────────
say "4/6 제출 파일 생성"
SUB="out/submission_${TAG}.csv"; MAJ="out/submission_${TAG}_maj.csv"
python3 src/submit_fuse.py --gen "$GEN" --orm "$OSC" --ref "$CSV" \
  --gamma "$GAMMA" --how mean --out "$SUB" || die "융합 실패"
python3 src/submit_fuse.py --gen "$GEN" --ref "$CSV" --out "$MAJ" \
  || echo "  (주의: 다수결 폴백 생성 실패 — 융합본만 사용)"
ok "$SUB"; [ -f "$MAJ" ] && ok "$MAJ (폴백)"

# ─────────────────────────── [5] 자체 검증 ───────────────────────────
say "5/6 자체 검증"
python3 - "$SUB" "$MAJ" "$CSV" <<'PY' || die "검증 실패 —— 이 파일을 제출하지 마라"
import sys, pandas as pd
sub, maj, ref = sys.argv[1], sys.argv[2], sys.argv[3]
r = pd.read_csv(ref); s = pd.read_csv(sub)
bad = []
if len(s) != len(r): bad.append(f"행 수 {len(s)} ≠ 입력 {len(r)}")
ac = [c for c in s.columns if c.lower() in ("answer","prediction","pred","label")]
if not ac: bad.append(f"정답 컬럼 없음: {list(s.columns)}")
else:
    col = s[ac[0]]
    if col.isna().any(): bad.append(f"결측 {int(col.isna().sum())}건")
    nonint = sum(1 for v in col.dropna()
                 if not (isinstance(v,(int,)) or (str(v).lstrip('-').isdigit())))
    if nonint: bad.append(f"정수 아님 {nonint}건")
try:
    m = pd.read_csv(maj)
    if len(m) == len(s):
        flip = int((s[ac[0]].astype(str).values != m[ac[0]].astype(str).values).sum())
        pct = flip/len(s)*100
        print(f"  검증기가 다수결을 뒤집은 문항: {flip} ({pct:.1f}%)")
        if flip == 0: bad.append("뒤집힘 0건 —— ORM 점수가 반영되지 않았다")
        if pct > 25:  bad.append(f"뒤집힘 {pct:.1f}% > 25% —— 비정상")
except Exception: print("  (다수결 대조 생략)")
if bad:
    print("\n  \033[1;31m문제:\033[0m"); [print("   -", b) for b in bad]; sys.exit(1)
print(f"  행 {len(s)} · 결측 0 · 전부 정수")
PY
ok "검증 통과"

# ─────────────────────────── [6] 요약 ───────────────────────────
say "6/6 요약"
EL=$(( ($(date +%s) - T_START) / 60 ))
echo "  총 소요 ${EL}분 · 문항 $N · 문항당 $(python3 -c "print(f'{$EL*60/$N:.2f}')")초"
echo "  제출: $SUB"
echo "  폴백: $MAJ"
echo -e "\n\033[1;32m완료. 위 파일을 제출하면 된다.\033[0m"
