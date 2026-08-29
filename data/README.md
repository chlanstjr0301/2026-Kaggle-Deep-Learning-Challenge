# 데이터 배치 안내

대회 제공 데이터는 재배포하지 않는다. 대회 페이지에서 직접 내려받아 이 폴더에 둔다.

| 파일 | 출처 | 용도 |
|---|---|---|
| `deep_chal_math_dataset_train.csv` | 대회 제공 | ORM 학습 데이터 생성, 내부 검증셋 분리 |
| `deep_chal_math_dataset_leaderboard_filtered.csv` | 대회 제공 (2026-08-03 정제판) | 리더보드 추론 입력 |
| `deep_chal_math_dataset_test.csv` | 대회 제공 (2026-08-31 공개) | 최종 추론 입력 |
| `train_filtered_ids.csv` | 운영진 제공 | 학습 데이터에서 제외할 627문항 |

필드는 `id` · `question` · `answer` 세 개다. `answer`는 정수이며 평가셋에서는 비어 있다.

외부 공개 데이터셋은 최종 구성에 포함되지 않는다. 탐색 과정의 사용 내역은
저장소 루트 `COMPLIANCE.md` §1에 명시했다.
