#!/usr/bin/env python3
"""gpt-oss-20b 反量化加载冒烟（供 run_gpt_oss_a100.sh 的 smoke 步调用）。

单独成文件而非用 heredoc：本环境下 `conda run + heredoc(<<PY)` 会吞掉子进程
stdout，导致看起来"无输出"（实际已加载成功）。用文件 + flush 保证逐行可见。

用法：python _smoke_load.py [MODEL_DIR]   （默认 ./gpt-oss-20b）
"""

import sys
import traceback

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "./gpt-oss-20b"

print("PYSTART", flush=True)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    print("transformers import OK", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print("tokenizer OK vocab", tok.vocab_size, flush=True)
    print("loading model (dequantize=True)... 这步慢，耐心", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=Mxfp4Config(dequantize=True),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    m.eval()
    print("MODEL LOADED", type(m).__name__, "dtype", next(m.parameters()).dtype, flush=True)
    print("gpu mem GB", round(torch.cuda.memory_allocated() / 1024**3, 1), flush=True)
    ids = tok("The capital of France is", return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m(**ids)
    print("logits", tuple(out.logits.shape), "top1", repr(tok.decode(out.logits[0, -1].argmax())), flush=True)
    print("SMOKE_OK", flush=True)
except Exception:
    traceback.print_exc()
    print("SMOKE_FAIL", flush=True)
    sys.exit(1)
