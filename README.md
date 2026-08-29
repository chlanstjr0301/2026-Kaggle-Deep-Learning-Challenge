# 아주 소중한 딥러닝 챌린지 2026 — 제출 저장소

**참가자**: 최문석 (경북대학교, 개인 참가) · chlanstjr0301@gmail.com

---

## 1. 한 줄 요약

**Qwen2.5-3B-Instruct 무수정 생성 k=64 + 동일 베이스 LoRA 검증기 융합 선별.**

```
생성   프롬프트 C · chat template · k=64 · T=0.8 · top_p 1.0 · max_tokens 3072
선별   s(a) = log n_a + 0.5 · z̄(a)      z̄ = 문항 내 표준화 ORM 점수의 표본 평균
동률   답 오름차순으로 결정적 처리
```

생성 모델에는 어떤 학습 가중치도 병합하지 않았다. 학습한 구성요소는 채점용 LoRA
어댑터 하나(`ckpt/orm_r2`)뿐이며, 같은 베이스 위에 얹는다.

리더보드 831문항 기준 **663/831 (0.79783)**. 검증기 미적용 순수 다수결은 653/831.

---

## 2. 규칙 준수

| 조항 | 요구 | 본 제출 |
|---|---|---|
| 4.1b · 4.3 | 베이스는 Qwen2.5-3B-Instruct 단독, 타 모델 가중치 병합 금지 | 생성·채점 모두 동일 베이스. 실행 시 shard SHA-256 2개로 검증 |
| 4.3 | 추론 시 외부 모델 앙상블 금지 | 동일 베이스 LoRA 어댑터만 사용 (운영진 2026-08-04 허용 확인) |
| 5.1b | test 문항을 학습에 사용 금지 | 미사용 |
| 5.2c | 외부 데이터셋 목록 명시 | `COMPLIANCE.md` §1 |
| 5.3a·b·c | 학습 데이터 구축용 API 허용, test 답 생성·검색 금지 | 외부 API는 train 문항에만 사용 |
| 6a | 추론 시 인터넷 차단 | `run_final.sh` 전 과정 오프라인 |
| 6c | Majority Voting · Best-of-N 허용 | maj@64 + 자체 검증기 |
| 운영진 2026-08-28 | 학습 과정에 타 모델이 풀이 적절성을 판정하는 개입 금지 | 해당 없음. `COMPLIANCE.md` §2.1 |

**ORM 라벨은 전부 주최 측 gold 정답과의 정수 일치로 생성한다**
(`src/build_orm_data.py`, `v == g`). 외부 모델이 풀이를 O/X 판정한 지점은 없다.

---

## 3. 환경

```
GPU        NVIDIA A100 SXM4 40GB × 1
디스크      100GB 이상
파이썬      3.12
```

추론과 학습의 `transformers` 요구 버전이 충돌하여 가상환경을 둘로 나눈다.

| 환경 | 용도 | 주요 패키지 |
|---|---|---|
| `/venv/main` | 추론 | vllm 0.27.1 · transformers 5.x |
| `/venv/train` | 학습·병합 | transformers 4.57.6 · peft 0.20.0 · trl 1.10.0 |

전체 목록은 `docs/pip_freeze.txt`, 하드웨어 실측은 `docs/ENVIRONMENT.txt`.

---

## 4. 재현 절차 — 추론

```bash
# 0) 베이스 모델과 LoRA 어댑터 내려받기
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./qwen3b
hf download choimunseok/dc2026-qwen3b-orm --local-dir ckpt
sha256sum -c ckpt/SHA256SUMS.txt          # 무결성 확인

# 1) LoRA 어댑터를 베이스에 병합
bash scripts/merge_orm.sh orm_r2          # -> ckpt/orm_r2-merged

# 2) 추론
bash scripts/run_final.sh <입력CSV> final  # -> out/submission_final.csv
```

입력 CSV의 필드는 `id` · `question` 두 개면 된다. `answer` 열은 없어도 되고,
있으면 무시한다.

2,000문항 기준 A100 40GB 1장에서 약 2시간 15분. 생성 처리량 약 10,000 tok/s.

`run_final.sh`는 6단계이며 각 단계에 사전점검이 있다.

| 단계 | 내용 |
|---|---|
| 0 | 입력·모델·ORM config·디스크(≥20G)·GPU 여유(≥30000MiB)·잔존 프로세스 확인 |
| 1 | 입력 정규화 |
| 2 | 생성 (k=64) |
| 3 | ORM 채점 |
| 4 | 융합 및 제출 파일 생성 |
| 5 | 자체 검증 — 다수결 대비 뒤집힘이 0이거나 25% 초과면 거부 |
| 6 | 요약 |

중단 시 체크포인트 재개가 가능하다.

---

## 5. 재현 절차 — 학습

```bash
bash scripts/ensure_train_venv.sh                       # 학습 환경 생성

python3 src/clean_data.py                               # 데이터 정제 + val 분리
python3 src/rft_sample.py --model ./qwen3b \
    --data out/train_clean.csv --k 16 --temp 0.8 --tag r1
python3 src/build_orm_data.py --gen out/gen_rft_r1.jsonl \
    --data out/train_clean.csv --out out/orm_r1.jsonl
python3 src/train_orm.py --data out/orm_r1.jsonl \
    --stance plain --lr 1e-4 --rank 16 --epochs 1 --out ckpt/orm_r2
```

ORM 학습 설정: LoRA rank 16 · alpha 32 · lr 1e-4 · 1 epoch ·
target `q,k,v,o,gate,up,down_proj` · 학습 행 26,034.

---

## 6. 사용 데이터셋

| 구분 | 데이터 | 최종 구성 포함 |
|---|---|---|
| 주최 측 제공 | `deep_chal_math_dataset_train.csv` | ORM 학습 데이터 생성에 사용 |
| 주최 측 제공 | leaderboard / test CSV | 추론 **입력으로만** 사용 |
| 외부 공개 | `nvidia/OpenMathInstruct-2`, `microsoft/orca-math-word-problems-200k` | **미포함** (탐색 후 폐기) |
| 상용 API | DeepSeek, NVIDIA NIM — train 문항 풀이 생성 | **미포함** (탐색 후 폐기) |

대회 제공 데이터는 재배포하지 않는다. `data/README.md`의 안내에 따라 대회
페이지에서 직접 내려받아 `data/`에 둔다.

전체 명세와 라이선스는 `COMPLIANCE.md`.

---

## 7. 결과

| 구성 | 리더보드 831문항 |
|---|---|
| 순수 다수결 maj@64 | 653 / 831 (0.78580) |
| **다수결 + ORM 융합** | **663 / 831 (0.79783)** |

융합 이득은 서로 다른 생성 3회에서 +10 · +7 · +10, 평균 **+9.0문항 (+0.0108)**.

---

## 8. 재현성 — 검증 전 반드시 읽을 것

**동일 입력·동일 시드로 재실행해도 출력은 비트 단위로 재현되지 않는다.**
vLLM이 요청을 동적으로 배치하며, 시드는 샘플링만 고정하고 배치 구성에 따른
부동소수점 누적 순서까지는 고정하지 못한다.

실측 3회:

| 반복 | 문항 | 답이 바뀐 수 | 정확도 변화 |
|---|---|---|---|
| 리더보드, 동일 머신 | 831 | 14 (1.7%) | 0.79783 → 0.79663 |
| 검증셋, 동일 머신 | 1000 | 42 (4.2%) | 0.8250 → 0.8230 |
| 리더보드, **머신 재구축** | 831 | 18 (2.2%) | 0.79663 → 0.79783 |

세 번째는 인스턴스를 파괴하고 새로 대여해 본 저장소 내용만으로 복원한 결과다.
PyTorch의 CUDA 빌드가 12.8에서 13.0으로 바뀌었다.

**개별 문항의 답은 실행마다 2~4% 바뀌지만 집계 정확도는 1문항 안에서 움직인다.**
뒤집힘이 양방향으로 일어나 상쇄되기 때문이다. 바뀐 18문항의 1–2위 득표 차이
중앙값은 2이고 최대 16이다. 전체 831문항의 중앙값은 59다. 즉 불안정성은
사실상 동점인 문항에만 존재한다.

> **정확도는 ±0.004 이내에서 재현된다고 본다. 비트 단위 일치를 기대해서는 안 된다.**

결정성이 보장되는 부분: LoRA 병합, 답 추출 파서, 다수결 동률 처리, 융합 점수 계산.
**비결정성은 생성 단계에만 존재한다.**

---

## 9. 파일 구성

```
src/                   학습·추론·평가 스크립트 29개
scripts/               실행 진입점
ckpt/README.md         가중치 위치와 검증 방법 (파일은 Hugging Face)
ckpt/SHA256SUMS.txt    가중치 무결성 해시
data/                  대회 데이터 배치 위치 (원본 미포함)
results/submissions/   리더보드 제출본
docs/                  환경 명세
COMPLIANCE.md          외부 데이터·API 사용 명세, 규칙 준수 근거
REPRODUCIBILITY.md     재현 환경 및 재현성 상세
METHOD.pdf             방법론 문서 (31쪽)
```
