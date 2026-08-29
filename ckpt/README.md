# 학습된 모델 가중치

LoRA 어댑터 파일이 각 114.3 MiB 로 GitHub 의 파일당 100 MB 상한을 넘어
Hugging Face Hub 에 둔다. 규정 8.2 의 "학습된 모델 가중치" 및 "접근 방법"에 해당한다.

```
https://huggingface.co/choimunseok/dc2026-qwen3b-orm
```

## 내려받기

```bash
pip install -U huggingface_hub
hf download choimunseok/dc2026-qwen3b-orm --local-dir ckpt
```

`huggingface_hub` 1.29 부터 CLI 이름이 `huggingface-cli` 에서 `hf` 로 바뀌었다.
구버전 환경이면 `huggingface-cli download ...` 를 쓴다.

받으면 이 폴더가 다음과 같이 된다.

```
ckpt/orm_r1/   adapter_config.json · adapter_model.safetensors · run_meta.json · README.md
ckpt/orm_r2/   같음
```

## 무결성 확인

```bash
sha256sum -c SHA256SUMS.txt
```

`SHA256SUMS.txt` 는 이 저장소에 포함돼 있다. Hugging Face 를 거치지 않고도
받은 가중치가 학습 산출물과 동일한지 검증할 수 있다.

| 파일 | SHA-256 앞 8자 |
|---|---|
| `orm_r1/adapter_model.safetensors` | `52abfa42` |
| `orm_r2/adapter_model.safetensors` | `026f34f0` |

## 두 어댑터의 차이

같은 학습 데이터 파일을 읽었고 **학습 시 판정 stance 만 다르다.**

| | orm_r1 | orm_r2 |
|---|---|---|
| stance | harsh | plain |
| 학습 행 | 26,023 | 26,034 |
| lr / rank / alpha / epochs | 1e-4 / 16 / 32 / 1 | 동일 |
| 최종 채택 | 아니오 | **예** |

`run_final.sh` 는 `ckpt/orm_r2-merged` 를 사용한다. 병합은
`bash scripts/merge_orm.sh orm_r2` 로 수행한다.

## 모델 카드에 관한 주석

`orm_r1/README.md` 와 `orm_r2/README.md` 는 peft 가 자동 생성한 모델 카드다.
프론트매터의 `base_model` 이 학습 당시 로컬 경로(`./qwen3b`)로 적혀 있어
Hugging Face 업로드가 거부되었고, 이를 `Qwen/Qwen2.5-3B-Instruct` 로 바꾸었다.
**가중치와 `adapter_config.json` 은 손대지 않았다** —— 위 표의 SHA-256 이 그 증거다.

## dtype

어댑터는 F32 로 저장돼 있다. 병합 시 bfloat16 으로 캐스팅된다.
용량을 줄이려고 bf16 으로 재저장하지 않았다 —— 배포 가중치를 학습 산출물과
비트 단위로 같게 유지하기 위함이다.
