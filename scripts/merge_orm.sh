#!/bin/bash
# LoRA 어댑터를 베이스에 병합한다.
#
#   bash scripts/merge_orm.sh orm_r2
#   bash scripts/merge_orm.sh orm_r1
#
# 토크나이저는 베이스에서 가져온다. LoRA 는 어휘를 바꾸지 않으므로
# (modules_to_save=None, vocab_size 151936 동일) 베이스의 것이 정확하다.
set -e
NAME=${1:-orm_r2}
SRC=ckpt/$NAME
DST=ckpt/$NAME-merged
PY=${PYTHON:-/venv/train/bin/python}

[ -f "$SRC/adapter_config.json" ] || { echo "✗ 어댑터 없음: $SRC"; exit 1; }
[ -f ./qwen3b/config.json ]       || { echo "✗ 베이스 없음: ./qwen3b"; exit 1; }
[ -x "$PY" ] || PY=python3

"$PY" - "$SRC" "$DST" <<'PYEOF'
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
src, dst = sys.argv[1], sys.argv[2]
b = AutoModelForCausalLM.from_pretrained("./qwen3b", dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(b, src).merge_and_unload().to(torch.bfloat16)
m.save_pretrained(dst, safe_serialization=True)
AutoTokenizer.from_pretrained("./qwen3b").save_pretrained(dst)
print(f"[merge] {dst} 완료")
PYEOF

for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
  [ -f "$DST/$f" ] || { echo "✗ 누락: $DST/$f"; exit 1; }
done
echo "✓ 병합 검증 통과: $DST"
