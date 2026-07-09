#!/usr/bin/env bash
# gpt-oss-20b 在 A100（无 FP4 硬件）上跑 J-Space 的端到端脚本。
#
# 前置：已建 conda env `gptoss`（clone 自 StarPhoton + transformers>=5 + kernels），
# torch2.6/triton3.2 未动。模型仓库已 git clone 到 ./gpt-oss-20b（权重当前可能仍是
# LFS 指针）。本脚本负责：拉权重 → 反量化加载冒烟 → J-Space 全流程 → g 校验。
#
# 用法：
#   bash run_gpt_oss_a100.sh            # 全流程
#   STEP=download bash run_gpt_oss_a100.sh   # 只跑某一步：download|smoke|dict|readout|decompose|steer|gcheck
#
# 说明：MXFP4 专家权重加载时反量化为 bf16（A100 compute_cap 8.0 无 FP4 tensor core，
# 且 J-lens 求 VJP 需稠密可导权重）。反量化后显存约 40GB，A100 80GB 足够。

set -euo pipefail

ENV=gptoss
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HERE/gpt-oss-20b}"
OUT="${OUT:-$HERE/gpt_oss_20b_jspace_dictionary.pt}"
LAYERS="${LAYERS:-0,4,8,12,16,20}"
LAYER="${LAYER:-12}"
STEP="${STEP:-all}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

run() { echo -e "\n\033[1;36m==> $*\033[0m"; conda run -n "$ENV" "$@"; }

# gpt-oss 用 HF Xet 存储后端，snapshot_download 需 hf_xet 包否则静默跳过；
# 这里改用 curl 直下真实数据流（走镜像，绕开 Xet/hub 所有坑），带 -C - 断点续传。
# 只拉 HF transformers 需要的文件；metal/ 与 original/ 不下。
BASE="$HF_ENDPOINT/openai/gpt-oss-20b/resolve/main"
DL_FILES=(
  "model-00000-of-00002.safetensors"
  "model-00001-of-00002.safetensors"
  "model-00002-of-00002.safetensors"
  "model.safetensors.index.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "special_tokens_map.json"
  "config.json"
  "generation_config.json"
  "chat_template.jinja"
)

is_pointer() {  # 判断文件是否仍是 LFS 指针（<1KB 且以 version https 开头）
  local f="$1"
  [ -f "$f" ] || return 0
  [ "$(stat -c%s "$f")" -lt 1024 ] && head -c 40 "$f" | grep -q '^version https' && return 0
  return 1
}

step_download() {
  echo -e "\n\033[1;36m==> 拉取 HF 权重（curl 直下，走镜像 $HF_ENDPOINT，跳过 metal/original）\033[0m"
  mkdir -p "$MODEL_DIR"
  for f in "${DL_FILES[@]}"; do
    tgt="$MODEL_DIR/$f"
    if [ -f "$tgt" ] && ! is_pointer "$tgt"; then
      echo "  [skip] $f 已存在（$(stat -c%s "$tgt") B）"
      continue
    fi
    # 指针文件先删，避免 -C - 在指针上续传
    is_pointer "$tgt" && rm -f "$tgt"
    echo "  [get ] $f"
    curl -fL -C - --retry 5 --retry-delay 3 "$BASE/$f" -o "$tgt"
  done
  echo "-- 校验权重分片已非指针 --"
  for f in model-00000-of-00002.safetensors model-00001-of-00002.safetensors \
           model-00002-of-00002.safetensors tokenizer.json; do
    t="$MODEL_DIR/$f"
    if is_pointer "$t"; then echo "  ✗ $f 仍是指针！"; else echo "  ✓ $f $(du -h "$t" | cut -f1)"; fi
  done
}

step_smoke() {
  echo -e "\n\033[1;36m==> 反量化加载冒烟（bf16, device_map=auto）\033[0m"
  # 用独立脚本文件而非 heredoc：本环境 conda run + heredoc 会吞子进程 stdout。
  # --no-capture-output 进一步确保 conda 不缓冲输出；期望末行打印 SMOKE_OK。
  HF_HUB_OFFLINE=1 conda run --no-capture-output -n "$ENV" python "$HERE/_smoke_load.py" "$MODEL_DIR"
}

# J-Space 各步：--local-files-only 避免联网，--dequantize-mxfp4 auto 会自动反量化
JS() { HF_HUB_OFFLINE=1 conda run --no-capture-output -n "$ENV" python "$HERE/jspace_gpt_oss.py" "$@" \
        --model-id "$MODEL_DIR" --local-files-only --torch-dtype bfloat16 --device-map auto; }

step_dict() {
  echo -e "\n\033[1;36m==> 构建 J-lens 字典（严格版：rows of W_U J_ℓ，对 pre-norm h_final 求 VJP）\033[0m"
  JS build-dictionary --prompts-file "$HERE/calibration_prompts.txt" \
     --candidates-file "$HERE/candidate_concepts.txt" \
     --layers "$LAYERS" --max-prompts 8 --max-length 128 --max-pairs 1 \
     --position-mode last --out "$OUT"
}

step_readout() {
  echo -e "\n\033[1;36m==> readout（默认点积口径）\033[0m"
  JS readout --dictionary "$OUT" --prompt "A spider builds a" \
     --layer "$LAYER" --position -1 --top-k 20 > "$HERE/readout_spider_layer${LAYER}.json"
  cat "$HERE/readout_spider_layer${LAYER}.json"
}

step_decompose() {
  echo -e "\n\033[1;36m==> decompose（稀疏非负 J-space 分解 k=25）\033[0m"
  JS decompose --dictionary "$OUT" --prompt "A spider builds a" \
     --layer "$LAYER" --position -1 --k 25 > "$HERE/decompose_spider_layer${LAYER}.json"
  cat "$HERE/decompose_spider_layer${LAYER}.json"
}

step_steer() {
  echo -e "\n\033[1;36m==> intervene steer（注入 spider 概念）\033[0m"
  JS intervene --dictionary "$OUT" --prompt "The animal crawled across the" \
     --layer "$LAYER" --position -1 --mode steer --token " spider" \
     --alpha 2.0 --steps 32 > "$HERE/steer_spider_layer${LAYER}.json"
  cat "$HERE/steer_spider_layer${LAYER}.json"
}

step_gcheck() {
  echo -e "\n\033[1;36m==> g 折入校验（回答 h_final 是否 pre-norm + 折 g 排序偏差）\033[0m"
  HF_HUB_OFFLINE=1 conda run --no-capture-output -n "$ENV" python "$HERE/check_gnorm_alignment.py" \
     --model-id "$MODEL_DIR" --layer "$LAYER" --torch-dtype bfloat16 --device-map auto --local-files-only
}

case "$STEP" in
  download) step_download ;;
  smoke)    step_smoke ;;
  dict)     step_dict ;;
  readout)  step_readout ;;
  decompose) step_decompose ;;
  steer)    step_steer ;;
  gcheck)   step_gcheck ;;
  all)
    step_download
    step_smoke
    step_dict
    step_readout
    step_decompose
    step_steer
    step_gcheck
    echo -e "\n\033[1;32m全流程完成。产物：$OUT, readout/decompose/steer_*.json\033[0m"
    ;;
  *) echo "未知 STEP=$STEP（可选 download|smoke|dict|readout|decompose|steer|gcheck|all）"; exit 1 ;;
esac
