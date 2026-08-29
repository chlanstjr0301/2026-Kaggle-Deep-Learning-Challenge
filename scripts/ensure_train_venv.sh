#!/bin/bash
# /venv/train 이 없으면 재생성한다. 있으면 검증만 하고 즉시 통과.
if /venv/train/bin/python -c "import transformers,peft,trl,inspect;assert 'warmup_ratio' in inspect.signature(transformers.TrainingArguments.__init__).parameters" 2>/dev/null; then
  echo "[venv] train 정상 · tf $(/venv/train/bin/python -c 'import transformers;print(transformers.__version__)')"
  exit 0
fi
echo "[venv] train 없음/손상 → 재생성"
rm -rf /venv/train
python3 -m venv --system-site-packages /venv/train
/venv/train/bin/pip install -q "transformers==4.57.6" "peft==0.20.0" "trl==1.10.0" "accelerate==1.14.0" "datasets==5.0.1"
/venv/train/bin/python -c "import transformers;print('[venv] 복원 완료 tf', transformers.__version__)"
