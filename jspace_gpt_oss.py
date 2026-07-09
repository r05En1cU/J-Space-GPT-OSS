#!/usr/bin/env python3
"""
面向 GPT-OSS-20B 一类 HuggingFace decoder-only 语言模型的
J-Space / Jacobian Lens（雅可比透镜）实用复现实现。

论文来源：https://transformer-circuits.pub/2026/workspace/index.html

论文算法定义了从中间层 residual stream h_ℓ 到「最终 residual 流」的平均
Jacobian J_ℓ。这里的最终 residual 流严格指**最后一个 decoder block 的输出残差流
（final-norm 之前）**，final-norm 不包含在 J_ℓ 内：

    J_ℓ = E_{t, t'≥t, prompt}[ ∂h_final,t' / ∂h_ℓ,t ]

对 20B 级模型不物化完整的 d_model × d_model Jacobian，而是构造等价的
token 向量字典。每个 token 的 J-lens 向量取为 W_U·diag(g)·J_ℓ 的行：

    v_{ℓ, token} = J_ℓ^T · diag(g) · W_U[token]   （即 rows of W_U·diag(g)·J_ℓ）

其中 g 是 final-norm（RMSNorm）的逐通道可学习增益。实现上通过一次 VJP 得到：
在第 ℓ 层注入 leaf，在最后一个 block 输出处捕获 pre-norm 的最终残差流 h_final
（仍与 leaf 相连），对标量 `W_U[token] · (g ⊙ h_final[target])` 关于 leaf 求梯度，
取 source 位置的分量并跨 prompt/位置对求平均。

关于 g 的折入（2026-07-09 真机校验后定案）：论文说 readout 探针 ⟨v_t,h_ℓ⟩
「up to a data-dependent normalization factor」等于 pre-softmax logit——该因子指
RMSNorm 的 1/rms（标量、每位置相同、不改排序）。但 g 是**逐通道 diag(g) 而非标量**，
若不折入，readout 排序会系统性偏离模型真实 logit（gpt-oss-20b 上 raw vs 折 g 的
Spearman 仅 0.705、top-10 只重合 5/10）。故把 diag(g) 折入有效解嵌 W_U_eff =
W_U·diag(g)；J_ℓ 仍严格到 pre-norm 残差不变。模型无可学习 g 时退化为 raw W_U。

该 token 向量字典即可支持 readout、稀疏非负分解、steering、ablation 与
coordinate patching，而无需保存完整平均 Jacobian 矩阵。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - 依赖缺失保护
    raise SystemExit(
        "缺少依赖：transformers。请用 `pip install transformers accelerate` 安装。"
    ) from exc


DEFAULT_MODEL_ID = "openai/gpt-oss-20b"
FORMAT_VERSION = 1
EPS = 1e-8
BF16_REL_EPS = 2.0 ** -8
# Primary readout-full regression tolerance: 0.125 = 32 bf16 relative eps.
# This is below the loose sqrt(d_model) accumulation envelope for GPT-OSS-20B
# (2^-8 * sqrt(2880) ~= 0.21), but far below scale/offset/g/transpose bugs.
READOUT_FULL_RELERR_THRESHOLD = 0.125
READOUT_FULL_SWAP_ULP_FACTOR = 3.0
READOUT_FULL_TOPK_GATE = 10


class RegressionGateError(RuntimeError):
    """Raised when readout-full dictionary regression fails, carrying the gate result."""

    def __init__(self, message: str, result: Mapping[str, Any]):
        super().__init__(message)
        self.result = dict(result)


DEFAULT_PROMPTS = [
    "The capital of France is",
    "A spider builds a web because",
    "In quantum mechanics, a particle can",
    "The recipe starts by chopping onions and",
    "When a patient has a fever, the doctor",
    "A compiler transforms source code into",
    "The ocean tide rises when",
    "To solve the equation, first isolate",
]


DEFAULT_CANDIDATES = [
    " Paris", " France", " capital", " city", " Europe",
    " spider", " web", " insect", " animal", " legs",
    " quantum", " particle", " wave", " energy", " physics",
    " doctor", " fever", " patient", " medicine", " hospital",
    " code", " compiler", " program", " function", " variable",
    " equation", " solve", " number", " matrix", " proof",
    " memory", " attention", " token", " language", " concept",
    " yes", " no", " true", " false", " because",
]


COMMON_BLOCK_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "transformer.blocks",
    "backbone.layers",
    "decoder.layers",
)


@dataclass
class BuildConfig:
    model_id: str = DEFAULT_MODEL_ID
    layers: str = "all"
    max_length: int = 128
    max_prompts: Optional[int] = None
    max_pairs: int = 1
    position_mode: str = "last"  # last | all-same | causal-window（源/目标位置对的采样模式）
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    local_files_only: bool = False
    normalize_saved_vectors: bool = False


@dataclass
class DecompositionResult:
    active_indices: List[int]
    active_token_ids: List[int]
    active_token_texts: List[str]
    coeffs: List[float]
    residual_norm: float
    target_norm: float
    explained_fraction: float


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def parse_dtype(name: str) -> Optional[torch.dtype]:
    name = (name or "auto").lower()
    if name in {"auto", "none"}:
        return None
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in aliases:
        raise ValueError(f"不支持的 dtype {name!r}；请用 auto/fp32/fp16/bf16。")
    return aliases[name]


def get_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(path)
        cur = getattr(cur, part)
    return cur


def find_decoder_blocks(model: torch.nn.Module) -> Tuple[Sequence[torch.nn.Module], str]:
    for path in COMMON_BLOCK_PATHS:
        try:
            blocks = get_attr_path(model, path)
        except AttributeError:
            continue
        if isinstance(blocks, (torch.nn.ModuleList, list, tuple)) and len(blocks) > 0:
            return blocks, path

    # 保守回退：选取含 transformer 风格 block 的最大 ModuleList。
    best_name = None
    best_module = None
    best_len = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > best_len:
            first = module[0] if len(module) else None
            first_name = first.__class__.__name__.lower() if first is not None else ""
            if any(key in first_name for key in ("block", "layer", "decoder")):
                best_name, best_module, best_len = name, module, len(module)
    if best_module is not None:
        return best_module, best_name or "<modulelist>"

    raise RuntimeError(
        "无法定位 decoder blocks。请把该模型的 block 路径加入 COMMON_BLOCK_PATHS。"
    )


# unembedding 之前的最终 norm（final-norm）的常见命名路径。
COMMON_FINAL_NORM_PATHS = (
    "model.norm",
    "model.final_layernorm",
    "model.final_layer_norm",
    "transformer.ln_f",
    "gpt_neox.final_layer_norm",
    "model.decoder.final_layer_norm",
)


def find_final_norm_weight(model: torch.nn.Module) -> Optional[Tensor]:
    """定位 unembedding 之前的 final-norm，返回其逐通道可学习增益 g（weight）。

    用于把 diag(g) 折入 J-lens 的有效解嵌（见 estimate_token_vectors_for_layer）。
    找不到 norm 层、或该 norm 无仿射 weight 时返回 None（调用方退化为 raw W_U）。

    ⚠️ 仅对 RMSNorm（如 gpt-oss、Llama 系）精确——RMSNorm 无均值中心化、无 bias，
    有效解嵌恰为 W_U·diag(g)。对 GPT-2 式**真 LayerNorm**（transformer.ln_f），只折
    weight 会漏掉：①均值中心化（等价对 W_U 各行去均值）②bias（W_U·bias 是 token 相关
    常数，会改变 readout 跨 token 排序）——此时本函数只做了「部分修正」。若检测到
    final-norm 带 bias，打印警告提示口径不完整。
    """
    for path in COMMON_FINAL_NORM_PATHS:
        try:
            module = get_attr_path(model, path)
        except AttributeError:
            continue
        weight = getattr(module, "weight", None)
        if weight is not None:
            if getattr(module, "bias", None) is not None:
                eprint(
                    f"[j-lens] 警告：final-norm({path}) 带 bias，疑似真 LayerNorm；"
                    "当前只折入 diag(g)，未折 bias 与均值中心化，readout 口径可能不完整。"
                )
            return weight.detach()
    return None


def infer_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def move_batch_to_device(batch: Mapping[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def replace_hidden_in_block_output(output: Any, new_hidden: Tensor) -> Any:
    if torch.is_tensor(output):
        return new_hidden
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    if isinstance(output, list):
        return [new_hidden] + list(output[1:])
    if isinstance(output, MutableMapping):
        copied = dict(output)
        for key in ("hidden_states", "last_hidden_state"):
            if key in copied:
                copied[key] = new_hidden
                return copied
    raise TypeError(f"Unsupported block output type for hook replacement: {type(output)!r}")


def first_hidden_from_block_output(output: Any) -> Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Block output tuple/list does not start with a tensor.")
        return output[0]
    if isinstance(output, Mapping):
        for key in ("hidden_states", "last_hidden_state"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Unsupported block output type: {type(output)!r}")


def normalize_position(pos: int, seq_len: int) -> int:
    if pos < 0:
        pos = seq_len + pos
    if pos < 0 or pos >= seq_len:
        raise IndexError(f"position {pos} out of range for sequence length {seq_len}")
    return pos


def parse_layers(spec: str, n_layers: int) -> List[int]:
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return list(range(n_layers))
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part or "-" in part:
            sep = ":" if ":" in part else "-"
            a, b = part.split(sep, 1)
            start = int(a)
            stop = int(b)
            if sep == "-":
                stop += 1
            layers.extend(range(start, stop))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    bad = [l for l in layers if l < 0 or l >= n_layers]
    if bad:
        raise ValueError(f"Layer(s) out of range 0..{n_layers - 1}: {bad}")
    return layers


def load_lines(path: Optional[str], defaults: Sequence[str]) -> List[str]:
    if path is None:
        return list(defaults)
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.strip("\n")
        if text.strip() and not text.lstrip().startswith("#"):
            lines.append(text)
    return lines


def calibration_prompts_sha256(prompts: Sequence[str]) -> str:
    """Stable hash for the exact calibration prompt list used by J-lens averaging."""
    blob = json.dumps(list(prompts), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def calibration_prompts_metadata(prompts: Sequence[str]) -> Dict[str, Any]:
    prompt_list = [str(p) for p in prompts]
    return {
        "prompts": prompt_list,
        "prompts_sha256": calibration_prompts_sha256(prompt_list),
        "num_prompts": len(prompt_list),
    }


def assert_same_calibration_prompts(dictionary_payload: Mapping[str, Any], prompts: Sequence[str]) -> Dict[str, Any]:
    """Require readout-full verification to use the exact dictionary calibration prompt set."""
    meta = dictionary_payload.get("calibration_prompts")
    if not isinstance(meta, Mapping) or "prompts_sha256" not in meta or "prompts" not in meta:
        raise ValueError(
            "该字典未记录 calibration 集,无法保证门有效,请用新代码重建字典。"
        )
    current = calibration_prompts_metadata(prompts)
    dict_hash = str(meta["prompts_sha256"])
    current_hash = str(current["prompts_sha256"])
    dict_count = int(meta.get("num_prompts", len(meta.get("prompts", []))))
    current_count = int(current["num_prompts"])
    if dict_hash != current_hash:
        raise ValueError(
            "calibration prompt 集不一致,拒绝对拍两个不同平均算子: "
            f"dictionary_sha256={dict_hash} dictionary_count={dict_count}; "
            f"current_sha256={current_hash} current_count={current_count}。"
        )
    if dict_count != current_count:
        raise ValueError(
            "calibration prompt 条数不一致,拒绝对拍两个不同平均算子: "
            f"dictionary_sha256={dict_hash} dictionary_count={dict_count}; "
            f"current_sha256={current_hash} current_count={current_count}。"
        )
    return current


def resolve_candidate_token_ids(tokenizer: Any, candidate_texts: Sequence[str]) -> Tuple[List[int], Dict[int, List[str]]]:
    token_to_texts: Dict[int, List[str]] = {}
    for text in candidate_texts:
        variants = [text]
        if text and not text.startswith(" "):
            variants.append(" " + text)
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            # J-space 是 token 级的。多 token 字符串取其最后一个 token，
            # 对 GPT 风格 BPE/SentencePiece 而言它通常是语义载体。
            token_id = int(ids[-1])
            token_to_texts.setdefault(token_id, [])
            if variant not in token_to_texts[token_id]:
                token_to_texts[token_id].append(variant)
    return sorted(token_to_texts), token_to_texts


def token_label(tokenizer: Any, token_id: int, token_to_texts: Optional[Mapping[int, Sequence[str]]] = None) -> str:
    if token_to_texts and token_id in token_to_texts and token_to_texts[token_id]:
        return token_to_texts[token_id][0]
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(token_id)


def make_position_pairs(seq_len: int, mode: str, max_pairs: int) -> List[Tuple[int, int]]:
    mode = mode.lower()
    max_pairs = max(1, int(max_pairs))
    if mode == "last":
        return [(seq_len - 1, seq_len - 1)]
    if mode == "all-same":
        stride = max(1, math.ceil(seq_len / max_pairs))
        positions = list(range(0, seq_len, stride))[:max_pairs]
        return [(p, p) for p in positions]
    if mode == "causal-window":
        pairs: List[Tuple[int, int]] = []
        stride = max(1, math.ceil(seq_len / max_pairs))
        for target in range(0, seq_len, stride):
            source = max(0, target - 1)
            pairs.append((source, target))
            if len(pairs) >= max_pairs:
                break
        return pairs
    raise ValueError("position_mode must be last, all-same, or causal-window")


def _config_is_mxfp4(model_id: str, trust_remote_code: bool, local_files_only: bool) -> bool:
    """探测模型 config 是否为 MXFP4 量化（gpt-oss 系列即是）。

    只读 config，不加载权重；探测失败按 False 处理（不影响非量化模型）。
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, local_files_only=local_files_only
        )
        qc = getattr(cfg, "quantization_config", None)
        if isinstance(qc, Mapping):
            method = qc.get("quant_method")
        else:
            method = getattr(qc, "quant_method", None)
        return str(method).lower() == "mxfp4"
    except Exception:
        return False


def load_model_and_tokenizer(
    model_id: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    dequantize_mxfp4: str = "auto",
) -> Tuple[Any, Any]:
    dtype = parse_dtype(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if device_map and device_map.lower() != "none":
        kwargs["device_map"] = device_map
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - 可选依赖保护
            raise SystemExit("--load-in-4bit 需要 bitsandbytes 及较新的 transformers。") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # MXFP4（gpt-oss 系列）反量化：J-lens 需要对 residual stream 做 VJP 求梯度，
    # 必须以 bf16 稠密权重加载；且非 Blackwell GPU（如 A100，compute_cap<10.0）
    # 无 FP4 硬件，也只能走反量化路径。
    # dequantize_mxfp4: "auto"（探测到 mxfp4 config 即开启）| "on"（强制）| "off"（关闭）。
    mode = (dequantize_mxfp4 or "auto").lower()
    if not load_in_4bit and mode != "off":
        want = mode == "on" or (
            mode == "auto"
            and _config_is_mxfp4(model_id, trust_remote_code, local_files_only)
        )
        if want:
            try:
                from transformers import Mxfp4Config
            except ImportError as exc:  # pragma: no cover - 依赖保护
                raise SystemExit(
                    "MXFP4 反量化需要 transformers>=4.55（含 Mxfp4Config）。"
                    "当前环境无 Mxfp4Config，请升级 transformers。"
                ) from exc
            kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            eprint(f"[mxfp4] 以反量化模式加载（dequantize=True, mode={mode}）")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # 我们只需要对 hook 注入的 residual stream leaf 求梯度。
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def encode_prompt(tokenizer: Any, prompt: str, max_length: int, device: torch.device) -> Dict[str, Tensor]:
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return move_batch_to_device(batch, device)


def forward_with_layer_leaf(
    model: Any,
    blocks: Sequence[torch.nn.Module],
    layer_idx: int,
    batch: Mapping[str, Tensor],
) -> Tuple[Any, Tensor, Tensor]:
    """前向一次，同时返回 (outputs, leaf, h_final)。

    - leaf：在第 layer_idx 层 block 输出处注入的、detach 后重新 requires_grad 的
      leaf 张量，作为 VJP 的求导变量（下游计算都从它重算）。
    - h_final：最后一个 decoder block（blocks[-1]）输出的隐藏态，即**最终层
      residual stream（final-norm 之前）**。它在计算图上仍与 leaf 相连（不 detach），
      供严格版 J_ℓ 的 VJP 使用。

    当 layer_idx 恰为最后一层时，leaf hook 与 h_final hook 落在同一 block 上；
    leaf hook 先注册、先执行并替换输出，h_final hook 后执行并捕获被替换后的输出，
    因此此时 h_final == leaf，梯度自然退化为 W_U[token]（对应 J_ℓ = I），无需特判。
    """
    captured: Dict[str, Tensor] = {}

    def leaf_hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        hidden = first_hidden_from_block_output(output)
        leaf = hidden.detach().requires_grad_(True)
        captured["leaf"] = leaf
        return replace_hidden_in_block_output(output, leaf)

    def final_hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        # 捕获 pre-norm 的最终残差流；不要 detach，保留其与 leaf 的连接。
        captured["h_final"] = first_hidden_from_block_output(output)
        return None

    # 先注册 leaf hook，再注册 final hook；若二者同在最后一层，final hook 会接收到
    # 被 leaf hook 替换后的输出，从而 h_final == leaf。
    handles = [
        blocks[layer_idx].register_forward_hook(leaf_hook),
        blocks[-1].register_forward_hook(final_hook),
    ]
    try:
        with torch.enable_grad():
            outputs = model(**batch, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if "leaf" not in captured:
        raise RuntimeError(f"Layer hook did not capture layer {layer_idx}")
    if "h_final" not in captured:
        raise RuntimeError("Final-block hook did not capture the pre-norm final residual stream")
    # 末层退化自检：ℓ 为最后一层时，final_hook 应收到 leaf_hook 替换后的输出，
    # 即 h_final 与 leaf 是同一张量（梯度退化为 W_U[token]，对应 J_ℓ = I）。
    # 该断言防止未来 PyTorch 版本改变「同一 module 多个 forward_hook 链式传递」
    # 的行为时静默出错。
    if layer_idx == len(blocks) - 1 and captured["h_final"] is not captured["leaf"]:
        raise RuntimeError(
            "末层退化断言失败：layer_idx 为最后一层时 h_final 应与 leaf 为同一张量；"
            "当前 PyTorch 的 forward_hook 链式行为可能已变化。"
        )
    return outputs, captured["leaf"], captured["h_final"]


def capture_layer_activation(
    model: Any,
    tokenizer: Any,
    blocks: Sequence[torch.nn.Module],
    prompt: str,
    layer_idx: int,
    position: int,
    max_length: int,
) -> Tensor:
    device = infer_input_device(model)
    batch = encode_prompt(tokenizer, prompt, max_length, device)
    captured: Dict[str, Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        hidden = first_hidden_from_block_output(output)
        pos = normalize_position(position, hidden.shape[1])
        captured["activation"] = hidden[0, pos].detach().float().cpu()
        return output

    handle = blocks[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(**batch, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if "activation" not in captured:
        raise RuntimeError(f"Layer hook did not capture activation for layer {layer_idx}")
    return captured["activation"]


def estimate_token_vectors_for_layer(
    model: Any,
    tokenizer: Any,
    blocks: Sequence[torch.nn.Module],
    prompts: Sequence[str],
    layer_idx: int,
    token_ids: Sequence[int],
    max_length: int,
    max_pairs: int,
    position_mode: str,
) -> Tuple[Tensor, Tensor]:
    """严格版：估计第 layer_idx 层的 J-lens token 向量 v_{ℓ,token} = rows of W_U_eff J_ℓ。

    对每个样本（prompt × 位置对），构造标量
        s = W_U[token] · (g ⊙ h_final[0, target_pos])
    其中 h_final 是 pre-norm 的最终残差流（final-norm 之前），g 是 final-norm 的
    逐通道可学习增益（diag(g)），W_U 是解嵌矩阵。然后对第 ℓ 层注入的 leaf 求梯度，
    取 grad[0, source_pos] 作为该样本的向量，跨 prompt/位置对求平均。这正是
    W_U·diag(g)·J_ℓ 的第 token 行（VJP 得到）。

    关于 g 的折入（2026-07-09 真机校验后定案，见 check_gnorm_alignment.py）：
    论文的 readout 探针 ⟨v_t,h_ℓ⟩「up to a data-dependent normalization factor」等于
    pre-softmax logit；该因子指 RMSNorm 的 1/rms（标量、每位置相同、不改排序）。但
    final-norm 的可学习增益 g 是**逐通道 diag(g) 而非标量**——真机上 raw（不折 g）
    与折 g 的 readout 排序 Spearman 仅 0.705、top-10 只重合 5/10，raw 口径系统性偏离
    模型真实 logit。因此把 g 折入有效解嵌 W_U_eff = W_U·diag(g)；J_ℓ 仍严格到
    pre-norm 残差不变，只是把 diag(g) 归入 readout 端的有效解嵌（与 logit-lens 折 LN
    增益的标准做法一致）。若模型 final-norm 无可学习 g（g=None），则退化为 raw W_U。

    MoE 注记：GPT-OSS 为 MoE 模型，注入 leaf 后下游 router 的 top-k 专家选择是
    离散不可导的，因此这里得到的 J_ℓ 实为「路由固定、仅经 gate softmax 与被选中
    专家权重」的局部雅可比，而非跨越专家切换的完整雅可比。这与论文 VJP 固定路由
    口径一致，属预期语义。
    """
    if not token_ids:
        raise ValueError("No candidate token ids provided.")
    # 取解嵌矩阵 W_U，形状 [vocab, d_model]。
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError(
            "model.get_output_embeddings() 返回 None，无法取得解嵌矩阵 W_U；"
            "严格版 J_ℓ 需要 W_U 来构造 W_U·h_final 标量。"
        )
    W_U = output_embeddings.weight  # [vocab, d_model]
    # final-norm 的逐通道增益 g（RMSNorm/LayerNorm 的 weight）；折入有效解嵌。
    g_weight = find_final_norm_weight(model)
    if g_weight is None:
        eprint("[j-lens] 未找到 final-norm 可学习增益 g，退化为 raw W_U（不折 g）。")
    device = infer_input_device(model)
    d_model: Optional[int] = None
    sums: Optional[Tensor] = None
    counts = torch.zeros(len(token_ids), dtype=torch.long)

    for prompt_idx, prompt in enumerate(prompts):
        batch = encode_prompt(tokenizer, prompt, max_length, device)
        seq_len = int(batch["input_ids"].shape[1])
        pairs = make_position_pairs(seq_len, position_mode, max_pairs)
        for pair_idx, (source_pos, target_pos) in enumerate(pairs):
            outputs, leaf, h_final = forward_with_layer_leaf(model, blocks, layer_idx, batch)
            if d_model is None:
                d_model = int(leaf.shape[-1])
                sums = torch.zeros((len(token_ids), d_model), dtype=torch.float32)
            source_pos = normalize_position(source_pos, leaf.shape[1])
            target_pos = normalize_position(target_pos, h_final.shape[1])
            # h_final 在 target 位置的向量；device_map=auto 下它与 W_U 可能不同 device。
            h_vec = h_final[0, target_pos]
            # 折入 final-norm 逐通道增益：h_eff = g ⊙ h_final（g 对齐到 h_vec 的 device/dtype）。
            if g_weight is not None:
                # 防御性检查：h_vec 应为 1D [d_model]，g 与之同形，避免意外广播。
                assert h_vec.dim() == 1 and g_weight.shape == h_vec.shape, (
                    f"g 折入维度不匹配：h_vec{tuple(h_vec.shape)} vs g{tuple(g_weight.shape)}"
                )
                h_eff = g_weight.to(device=h_vec.device, dtype=h_vec.dtype) * h_vec
            else:
                h_eff = h_vec
            for i, token_id in enumerate(token_ids):
                # W_U[token] 对齐到 h_eff 的 device/dtype，构造 pre-norm（已折 g）标量。
                w_row = W_U[int(token_id)].to(device=h_eff.device, dtype=h_eff.dtype)
                scalar = (w_row * h_eff).sum()
                grad = torch.autograd.grad(
                    scalar,
                    leaf,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                vec = grad[0, source_pos].detach().float().cpu()
                assert sums is not None
                sums[i].add_(vec)
                counts[i] += 1
            del outputs, leaf, h_final, h_vec
        eprint(
            f"layer={layer_idx} prompt={prompt_idx + 1}/{len(prompts)} "
            f"pairs={len(pairs)} vectors={len(token_ids)}"
        )
    if sums is None:
        raise RuntimeError("No vectors were estimated; check prompts and max_length.")
    denom = counts.clamp_min(1).float().unsqueeze(1)
    return sums / denom, counts


def score_activation(
    activation: Tensor,
    vectors: Tensor,
    token_ids: Sequence[int],
    token_to_texts: Optional[Mapping[int, Sequence[str]]] = None,
    tokenizer: Any = None,
    top_k: int = 20,
    cosine: bool = False,
) -> List[Dict[str, Any]]:
    """J-lens readout 打分：默认用点积 ⟨v_t, h_ℓ⟩。

    论文的 readout probe 就是 per-token 的点积 ⟨v_t, h_ℓ⟩，它「up to a
    data-dependent normalization factor」等于 pre-softmax logit（该因子对所有
    token 相同、不改变排序），因此这里默认返回原始点积分数。
    cosine=True 为可选路径（对 h 与每个向量做 L2 归一化后再点积），仅用于需要
    余弦相似度的场景，不是论文的默认 readout 语义。
    """
    h = activation.float().cpu()
    D = vectors.float().cpu()
    if cosine:
        h = F.normalize(h, dim=0)
        D = F.normalize(D, dim=1)
    scores = D @ h
    k = min(top_k, scores.numel())
    values, indices = torch.topk(scores, k=k)
    rows = []
    for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        token_id = int(token_ids[idx])
        rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "text": token_label(tokenizer, token_id, token_to_texts),
                "score": float(value),
            }
        )
    return rows


def rankdata_average(values: Tensor) -> Tensor:
    """Tie-aware 1-based average ranks, matching scipy.stats.rankdata(method='average')."""
    x = values.detach().float().cpu().flatten()
    n = int(x.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.float32)
    order = torch.argsort(x, stable=True)
    ranks = torch.empty(n, dtype=torch.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and bool(x[order[j]] == x[order[i]]):
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = float(avg_rank)
        i = j
    return ranks


def spearman_corr(a: Tensor, b: Tensor) -> float:
    """Dependency-free Spearman rho for regression gates."""
    if int(a.numel()) != int(b.numel()):
        raise ValueError(f"Spearman length mismatch: {a.numel()} vs {b.numel()}")
    if int(a.numel()) < 2:
        return float("nan")
    ra = rankdata_average(a)
    rb = rankdata_average(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.linalg.norm(ra) * torch.linalg.norm(rb)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float(((ra * rb).sum() / denom).item())


def resolve_single_token_id(tokenizer: Any, token: Optional[str], token_id: Optional[int]) -> int:
    """Resolve a user-facing token string/id to one token id, using existing GPT-style spacing."""
    if token_id is not None:
        return int(token_id)
    if token is None:
        raise ValueError("Provide token or token_id")
    text = token if token.startswith(" ") else " " + token
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Could not tokenize {token!r}")
    return int(ids[-1])


def resolve_tracked_token_ids(
    tokenizer: Any,
    track_tokens: Optional[Sequence[str]],
    track_token_ids: Optional[Sequence[int]],
) -> List[int]:
    seen = set()
    resolved: List[int] = []
    for token_id in track_token_ids or []:
        tid = int(token_id)
        if tid not in seen:
            seen.add(tid)
            resolved.append(tid)
    for token in track_tokens or []:
        tid = resolve_single_token_id(tokenizer, token, None)
        if tid not in seen:
            seen.add(tid)
            resolved.append(tid)
    return resolved


def rows_from_full_scores(
    scores: Tensor,
    tokenizer: Any,
    top_k: int,
    token_to_texts: Optional[Mapping[int, Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    """Format full-vocabulary top-k rows; ranks are true full-vocab ranks."""
    scores_cpu = scores.detach().float().cpu().flatten()
    k = min(int(top_k), int(scores_cpu.numel()))
    values, token_ids = torch.topk(scores_cpu, k=k)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "text": token_label(tokenizer, int(token_id), token_to_texts),
            "score": float(value),
        }
        for rank, (value, token_id) in enumerate(zip(values.tolist(), token_ids.tolist()), start=1)
    ]


def tracked_rows_from_scores(
    scores: Tensor,
    tokenizer: Any,
    token_ids: Sequence[int],
    token_to_texts: Optional[Mapping[int, Sequence[str]]] = None,
    prefix: str = "",
) -> List[Dict[str, Any]]:
    """Return exact full-vocab rank/score for selected token ids."""
    scores_cpu = scores.detach().float().cpu().flatten()
    rows: List[Dict[str, Any]] = []
    for token_id in token_ids:
        tid = int(token_id)
        if tid < 0 or tid >= int(scores_cpu.numel()):
            raise IndexError(f"token_id={tid} out of range for vocab size {scores_cpu.numel()}")
        score = float(scores_cpu[tid].item())
        rank = int((scores_cpu > scores_cpu[tid]).sum().item()) + 1
        key = f"{prefix}_" if prefix else ""
        rows.append(
            {
                "token_id": tid,
                "text": token_label(tokenizer, tid, token_to_texts),
                f"{key}rank": rank,
                f"{key}score": score,
            }
        )
    return rows


def random_token_sanity_from_scores(
    scores: Tensor,
    tokenizer: Any,
    exclude_token_ids: Sequence[int],
    n_random: int = 200,
    seed: int = 0,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Report deterministic random-token score sanity for full-vocab readout verification."""
    scores_cpu = scores.detach().float().cpu().flatten()
    vocab_size = int(scores_cpu.numel())
    excluded = {int(t) for t in exclude_token_ids if 0 <= int(t) < vocab_size}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected: List[int] = []
    # Oversample deterministically, filtering out dictionary candidates; fall back to a full scan if needed.
    for tid in torch.randperm(vocab_size, generator=generator).tolist():
        tid = int(tid)
        if tid not in excluded:
            selected.append(tid)
            if len(selected) >= int(n_random):
                break
    if not selected:
        return {"seed": int(seed), "n_random": 0, "note": "no non-dictionary token ids available"}
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    selected_scores = scores_cpu[selected_tensor]
    selected_order = torch.argsort(selected_scores, descending=True).tolist()
    top_rows = []
    for rank, local_idx in enumerate(selected_order[: min(int(top_k), len(selected))], start=1):
        tid = int(selected[local_idx])
        score = float(scores_cpu[tid].item())
        full_rank = int((scores_cpu > scores_cpu[tid]).sum().item()) + 1
        top_rows.append(
            {
                "rank_within_random_slice": rank,
                "full_vocab_rank": full_rank,
                "token_id": tid,
                "text": token_label(tokenizer, tid),
                "score": score,
            }
        )
    return {
        "seed": int(seed),
        "n_random": len(selected),
        "excluded_dictionary_tokens": len(excluded),
        "score_min": float(selected_scores.min().item()),
        "score_mean": float(selected_scores.mean().item()),
        "score_max": float(selected_scores.max().item()),
        "top_random_slice": top_rows,
    }


def score_full_vocab_from_vector(
    model: Any,
    readout_vector: Tensor,
    cosine: bool = False,
    chunk_size: int = 8192,
) -> Tensor:
    """Compute W_U @ readout_vector for the whole vocab without materializing extra copies.

    For the default dot-product readout this is the paper口径:
        scores = W_U @ (diag(g) · J_ℓ · h_ℓ)
    cosine=True is a diagnostic post-J-space cosine over W_U rows and the post-J vector;
    it is not used by the mandatory dictionary regression gate.
    """
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("model.get_output_embeddings() returned None; cannot score full vocab.")
    W_U = output_embeddings.weight.detach()
    vocab = int(W_U.shape[0])
    vec_cpu = readout_vector.detach().float().cpu().flatten()
    scores: List[Tensor] = []
    for start in range(0, vocab, int(chunk_size)):
        end = min(vocab, start + int(chunk_size))
        W_chunk = W_U[start:end].to(dtype=torch.float32)
        vec = vec_cpu.to(device=W_chunk.device, dtype=torch.float32)
        if cosine:
            chunk_scores = F.normalize(W_chunk, dim=1) @ F.normalize(vec, dim=0)
        else:
            chunk_scores = W_chunk @ vec
        scores.append(chunk_scores.detach().float().cpu())
    return torch.cat(scores, dim=0)


def apply_final_norm_gain_to_vector(vector: Tensor, g_weight: Optional[Tensor]) -> Tensor:
    """Apply the fixed 2026-07-09 diag(g) readout convention to one d_model vector."""
    out = vector.detach().float().cpu().flatten()
    if g_weight is None:
        return out
    g = g_weight.detach().float().cpu().flatten()
    if tuple(g.shape) != tuple(out.shape):
        raise ValueError(f"g/readout vector shape mismatch: g{tuple(g.shape)} vs vector{tuple(out.shape)}")
    return g * out


def patch_transformers_moe_grouped_mm_for_double_backward() -> Optional[Tuple[Any, Any]]:
    """Temporarily patch Transformers MoE grouped-mm for readout-full double-VJP JVP.

    On the current A100/torch stack, transformers.integrations.moe falls back to a
    custom autograd op whose backward uses torch.mm(..., out=...). That supports
    ordinary VJP (used by build-dictionary) but breaks the required double-backward
    JVP because out= kernels are not differentiable. For readout-full JVP only,
    replace the grouped-mm dispatcher with an equivalent Python matmul loop. The
    returned patch state must be passed to unpatch_transformers_moe_grouped_mm_for_double_backward,
    or use patched_transformers_moe_grouped_mm_for_double_backward() as a context manager.
    """
    try:
        import transformers.integrations.moe as moe
    except Exception as exc:  # pragma: no cover - optional Transformers internals
        eprint(f"[j-lens] MoE grouped-mm patch skipped: {exc}")
        return None

    current = getattr(moe, "_grouped_mm", None)
    if getattr(current, "_jspace_higher_order_patch", False):
        original = getattr(current, "_jspace_original_grouped_mm", None)
        if original is None:
            raise RuntimeError("MoE grouped-mm is patched but original dispatcher was not recorded.")
        return (moe, original)

    original = current

    def _grouped_mm_higher_order(input: Tensor, weight: Tensor, offs: Tensor) -> Tensor:
        chunks: List[Tensor] = []
        start = 0
        # offsets are routing metadata, not differentiable; the matmuls remain
        # differentiable w.r.t. input, which is all JVP needs because model
        # parameters are frozen in load_model_and_tokenizer().
        for i, end in enumerate(offs.detach().cpu().tolist()):
            end = int(end)
            if start < end:
                chunks.append(input[start:end] @ weight[int(i)])
            start = end
        if start < int(input.shape[0]):
            chunks.append(input.new_zeros((int(input.shape[0]) - start, int(weight.shape[-1]))))
        if not chunks:
            return input.new_zeros((int(input.shape[0]), int(weight.shape[-1])))
        return torch.cat(chunks, dim=0)

    setattr(_grouped_mm_higher_order, "_jspace_higher_order_patch", True)
    setattr(_grouped_mm_higher_order, "_jspace_original_grouped_mm", original)
    moe._grouped_mm = _grouped_mm_higher_order
    eprint("[j-lens] patched transformers MoE grouped_mm fallback for double-backward JVP")
    return (moe, original)


def unpatch_transformers_moe_grouped_mm_for_double_backward(patch_state: Optional[Tuple[Any, Any]]) -> bool:
    """Restore transformers.integrations.moe._grouped_mm after a scoped readout-full JVP."""
    if patch_state is None:
        return False
    moe, original = patch_state
    current = getattr(moe, "_grouped_mm", None)
    if getattr(current, "_jspace_higher_order_patch", False):
        moe._grouped_mm = original
        eprint("[j-lens] restored transformers MoE grouped_mm fallback after readout-full JVP")
        return True
    eprint("[j-lens] MoE grouped-mm patch not restored because dispatcher changed during JVP")
    return False


@contextmanager
def patched_transformers_moe_grouped_mm_for_double_backward() -> Iterator[bool]:
    """Scope the higher-order MoE patch to the enclosed readout-full JVP calculation."""
    patch_state = patch_transformers_moe_grouped_mm_for_double_backward()
    try:
        yield patch_state is not None
    finally:
        unpatch_transformers_moe_grouped_mm_for_double_backward(patch_state)


def estimate_average_jvp_for_layer(
    model: Any,
    tokenizer: Any,
    blocks: Sequence[torch.nn.Module],
    prompts: Sequence[str],
    layer_idx: int,
    tangent: Tensor,
    max_length: int,
    max_pairs: int,
    position_mode: str,
    jvp_dtype: str = "auto",
) -> Tuple[Tensor, int]:
    """Estimate mean_q[J_q · tangent] with the plan's double-VJP JVP trick.

    J_q maps the layer-ℓ residual stream leaf at source_pos to the final pre-norm
    residual stream at target_pos. The query tangent is the detached raw h_ℓ from
    capture_layer_activation, injected at each calibration source_pos. This exactly
    implements the approved average-Jacobian口径:
        mean_q[J_q · h_ℓ(query)]
    before diag(g) is applied in the readout vector.

    The Transformers MoE grouped-mm double-backward patch is scoped to this JVP
    calculation and restored before the function returns or raises.
    """
    if not prompts:
        raise ValueError("No calibration prompts provided for readout-full JVP.")
    device = infer_input_device(model)
    tangent_cpu = tangent.detach().float().cpu().flatten()
    jvp_torch_dtype = parse_dtype(jvp_dtype)
    summed: Optional[Tensor] = None
    count = 0

    with patched_transformers_moe_grouped_mm_for_double_backward():
        for prompt_idx, prompt in enumerate(prompts):
            batch = encode_prompt(tokenizer, prompt, max_length, device)
            seq_len = int(batch["input_ids"].shape[1])
            pairs = make_position_pairs(seq_len, position_mode, max_pairs)
            for source_pos, target_pos in pairs:
                outputs, leaf, h_final = forward_with_layer_leaf(model, blocks, layer_idx, batch)
                source_pos = normalize_position(source_pos, leaf.shape[1])
                target_pos = normalize_position(target_pos, h_final.shape[1])
                if int(leaf.shape[-1]) != int(tangent_cpu.numel()):
                    raise ValueError(
                        f"tangent dimension mismatch: leaf d={leaf.shape[-1]} vs tangent d={tangent_cpu.numel()}"
                    )

                # Double-backward JVP. First compute J^T u as a function of dummy u;
                # then differentiate that linear form w.r.t. u with grad_outputs=v.
                # This keeps the Jacobian direction as J·v (not J^T·v), matching PLAN_readout_full.md.
                y = h_final[0, target_pos]
                if jvp_torch_dtype is not None:
                    # Diagnostic dtype override only for y/u/tangent in the
                    # double-backward algebra. It does NOT lift model weights or
                    # matmul kernels, so it is not a valid high-precision reference
                    # for bf16/MXFP4 rounding questions by itself.
                    y = y.to(dtype=jvp_torch_dtype)
                u = torch.zeros_like(y, requires_grad=True)
                (Jt_u,) = torch.autograd.grad(
                    y,
                    leaf,
                    grad_outputs=u,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=False,
                )
                Jt_u_for_jvp = Jt_u.to(dtype=jvp_torch_dtype) if jvp_torch_dtype is not None else Jt_u
                v_at_source = torch.zeros_like(Jt_u_for_jvp)
                v_at_source[0, source_pos] = tangent_cpu.to(
                    device=Jt_u_for_jvp.device, dtype=Jt_u_for_jvp.dtype
                )
                (Jv,) = torch.autograd.grad(
                    Jt_u_for_jvp,
                    u,
                    grad_outputs=v_at_source,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                vec = Jv.detach().float().cpu().flatten()
                if summed is None:
                    summed = torch.zeros_like(vec)
                summed.add_(vec)
                count += 1
                del outputs, leaf, h_final, y, u, Jt_u, Jt_u_for_jvp, v_at_source, Jv
            eprint(
                f"readout-full layer={layer_idx} prompt={prompt_idx + 1}/{len(prompts)} "
                f"pairs={len(pairs)} jvp_count={count}"
            )

    if summed is None or count == 0:
        raise RuntimeError("No JVP samples were estimated; check prompts and position settings.")
    return summed / float(count), count


def verify_full_scores_against_dictionary(
    full_scores: Tensor,
    activation: Tensor,
    dictionary_payload: Mapping[str, Any],
    layer: int,
    tokenizer: Any,
) -> Dict[str, Any]:
    """Mandatory readout-full regression gate against the VJP dictionary.

    Primary criterion: max per-token absolute score error divided by dictionary
    score RMS over the candidate slice. This continuous numerical gate catches
    scale, offset, missing diag(g), transposed-J, and wrong-layer bugs that a rank
    statistic can hide. The fixed threshold is 0.125 (= 32 * 2^-8), chosen from
    bf16 relative precision with room for accumulated MXFP4-dequantized bf16 / MoE
    double-backward drift while staying far below gross implementation errors.

    Auxiliary rank criterion: top-10 set equality catches gross ordering bugs, but
    strict top-10 order is no longer a hard gate. Pairwise top-k inversions are
    classified by the larger A/B gap normalized by top-k score RMS: <= 3 bf16 ulps
    is a WARN-only sub-ULP near-tie, while larger inversions fail. The layer-16
    orbit/plant diagnostic established that such sub-ULP swaps can be benign after
    high-precision convergence and route equality checks; other layers receive the
    same exemption only through this explicit numerical near-tie test.
    """
    lp = layer_payload(dictionary_payload, layer)
    token_ids = [int(x) for x in lp["token_ids"]]
    if not token_ids:
        raise ValueError(f"Dictionary layer {layer} has no token_ids")
    dict_vectors = lp["vectors"].float().cpu()
    h = activation.detach().float().cpu()
    dict_scores = dict_vectors @ h
    token_index = torch.tensor(token_ids, dtype=torch.long)
    full_slice = full_scores.detach().float().cpu()[token_index]

    delta = full_slice - dict_scores
    abs_delta = delta.abs()
    score_rms = float(torch.sqrt(torch.mean(dict_scores.square())).item())
    score_scale = max(score_rms, EPS)
    rel_errors = abs_delta / score_scale
    max_rel_idx = int(torch.argmax(rel_errors).item())
    max_relative_error = float(rel_errors[max_rel_idx].item())
    mean_relative_error = float(rel_errors.mean().item())
    numeric_pass = bool(math.isfinite(max_relative_error) and max_relative_error <= READOUT_FULL_RELERR_THRESHOLD)

    rho = spearman_corr(full_slice, dict_scores)
    full_order = torch.argsort(full_slice, descending=True).tolist()
    dict_order = torch.argsort(dict_scores, descending=True).tolist()
    n_top = min(READOUT_FULL_TOPK_GATE, len(token_ids))
    full_top10 = [token_ids[i] for i in full_order[:n_top]]
    dict_top10 = [token_ids[i] for i in dict_order[:n_top]]
    same_order = full_top10 == dict_top10
    full_top10_set = set(full_top10)
    dict_top10_set = set(dict_top10)
    top10_set_match = full_top10_set == dict_top10_set
    intersection = sorted(full_top10_set.intersection(dict_top10_set))

    token_to_local_idx = {int(t): i for i, t in enumerate(token_ids)}
    full_rank = {int(token_ids[i]): r for r, i in enumerate(full_order)}
    dict_rank = {int(token_ids[i]): r for r, i in enumerate(dict_order)}
    inversion_tokens = sorted(full_top10_set.union(dict_top10_set), key=lambda t: min(full_rank[t], dict_rank[t]))
    if inversion_tokens:
        top_local = torch.tensor([token_to_local_idx[int(t)] for t in inversion_tokens], dtype=torch.long)
        top_rms = float(torch.sqrt(torch.mean(dict_scores[top_local].square())).item())
    else:
        top_rms = score_rms
    inversion_scale = max(top_rms, score_scale, EPS)
    sub_ulp_threshold = READOUT_FULL_SWAP_ULP_FACTOR * BF16_REL_EPS
    inversions: List[Dict[str, Any]] = []
    for left_pos, token_a in enumerate(inversion_tokens):
        for token_b in inversion_tokens[left_pos + 1 :]:
            # Rank order disagreement defines the swap. Equal tensor scores are still
            # ordered deterministically by argsort, so we classify by score gap below.
            if (full_rank[token_a] - full_rank[token_b]) * (dict_rank[token_a] - dict_rank[token_b]) >= 0:
                continue
            ia = token_to_local_idx[token_a]
            ib = token_to_local_idx[token_b]
            full_gap = float((full_slice[ia] - full_slice[ib]).item())
            dict_gap = float((dict_scores[ia] - dict_scores[ib]).item())
            gap_abs = max(abs(full_gap), abs(dict_gap))
            normalized_gap = gap_abs / inversion_scale
            classification = "sub_ulp_warn" if normalized_gap <= sub_ulp_threshold else "large_gap_fail"
            inversions.append(
                {
                    "token_a": {"token_id": int(token_a), "text": token_label(tokenizer, int(token_a))},
                    "token_b": {"token_id": int(token_b), "text": token_label(tokenizer, int(token_b))},
                    "full_rank_a": int(full_rank[token_a] + 1),
                    "full_rank_b": int(full_rank[token_b] + 1),
                    "dictionary_rank_a": int(dict_rank[token_a] + 1),
                    "dictionary_rank_b": int(dict_rank[token_b] + 1),
                    "full_gap_a_minus_b": full_gap,
                    "dictionary_gap_a_minus_b": dict_gap,
                    "gap_abs_over_topk_rms": float(normalized_gap),
                    "classification": classification,
                }
            )
    sub_ulp_inversions = [x for x in inversions if x["classification"] == "sub_ulp_warn"]
    large_gap_inversions = [x for x in inversions if x["classification"] == "large_gap_fail"]
    rank_pass = bool(top10_set_match and not large_gap_inversions)

    result = {
        "layer": int(layer),
        "score_rms_scale": float(score_scale),
        "relative_error_threshold": float(READOUT_FULL_RELERR_THRESHOLD),
        "relative_error_threshold_basis": "max(|A-B|) / rms(B_candidate_scores) <= 0.125 = 32 * 2^-8 bf16 rel eps; no max-min dynamic range",
        "max_relative_error": max_relative_error,
        "mean_relative_error": mean_relative_error,
        "max_abs_error": float(abs_delta.max().item()),
        "max_error_token": {
            "token_id": int(token_ids[max_rel_idx]),
            "text": token_label(tokenizer, int(token_ids[max_rel_idx])),
            "full_score": float(full_slice[max_rel_idx].item()),
            "dictionary_score": float(dict_scores[max_rel_idx].item()),
            "abs_error": float(abs_delta[max_rel_idx].item()),
            "relative_error": max_relative_error,
        },
        "numeric_pass": numeric_pass,
        "spearman": rho,
        "spearman_auxiliary_only": True,
        "top10_intersection_size": int(len(intersection)),
        "top10_n": int(n_top),
        "top10_set_match": bool(top10_set_match),
        "top10_same_order": bool(same_order),
        "topk_inversion_scale": float(inversion_scale),
        "sub_ulp_gap_threshold": float(sub_ulp_threshold),
        "sub_ulp_gap_threshold_basis": "3 * 2^-8 after normalization by top-k dictionary score RMS",
        "topk_inversions_total": int(len(inversions)),
        "topk_inversions_sub_ulp_warn": int(len(sub_ulp_inversions)),
        "topk_inversions_large_gap_fail": int(len(large_gap_inversions)),
        "topk_inversions": inversions,
        "rank_pass": rank_pass,
        "full_top10": [
            {"token_id": int(t), "text": token_label(tokenizer, int(t))} for t in full_top10
        ],
        "dictionary_top10": [
            {"token_id": int(t), "text": token_label(tokenizer, int(t))} for t in dict_top10
        ],
    }
    passed = bool(numeric_pass and rank_pass)
    result["passed"] = passed
    eprint(
        f"[verify] layer={layer} max_rel={max_relative_error:.6g}/{READOUT_FULL_RELERR_THRESHOLD:.6g} "
        f"Spearman={rho:.9f} top10_set={top10_set_match} same_order={same_order} "
        f"sub_ulp_swaps={len(sub_ulp_inversions)} large_swaps={len(large_gap_inversions)} pass={passed}"
    )
    if not passed:
        raise RegressionGateError(
            "readout-full regression gate failed: "
            f"layer={layer} max_rel={max_relative_error:.6g}/{READOUT_FULL_RELERR_THRESHOLD:.6g}, "
            f"top10_set_match={top10_set_match}, large_gap_inversions={len(large_gap_inversions)}, "
            f"sub_ulp_warn={len(sub_ulp_inversions)}. Check JVP direction/transpose/position/averaging/g口径.",
            result,
        )
    return result


def nnls_pgd(D_active: Tensor, target: Tensor, max_iter: int = 250, ridge: float = 1e-6) -> Tensor:
    """对活跃原子做小规模投影梯度 NNLS（非负最小二乘）求解。

    D_active: [m, d]，target: [d]。返回非负系数 coeffs [m] >= 0。
    """
    if D_active.numel() == 0:
        return torch.empty(0, dtype=target.dtype)
    D_active = D_active.float()
    target = target.float()
    gram = D_active @ D_active.T
    b = D_active @ target
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    try:
        x = torch.linalg.solve(gram + ridge * eye, b).clamp_min(0)
    except RuntimeError:
        x = torch.zeros_like(b)
    try:
        lip = float(torch.linalg.eigvalsh(gram + ridge * eye).max().item())
    except RuntimeError:
        lip = float(torch.linalg.matrix_norm(gram + ridge * eye, ord=2).item())
    step = 1.0 / max(lip, ridge, EPS)
    for _ in range(max_iter):
        grad = gram @ x - b + ridge * x
        new_x = (x - step * grad).clamp_min(0)
        if torch.max(torch.abs(new_x - x)).item() < 1e-7:
            x = new_x
            break
        x = new_x
    return x


def positive_sparse_pursuit(
    target: Tensor,
    dictionary: Tensor,
    token_ids: Sequence[int],
    token_texts: Optional[Mapping[int, Sequence[str]]] = None,
    tokenizer: Any = None,
    k: int = 25,
    tol: float = 1e-6,
    normalize_atoms: bool = True,
) -> Tuple[DecompositionResult, Tensor, Tensor]:
    """稀疏非负 gradient-pursuit 分解：min ||h - Σ a_i v_i||²，a_i≥0 且活跃数 ≤ k。

    注意与 readout 的区别：这里默认 normalize_atoms=True，即在做原子选择/拟合前
    对字典原子做 L2 归一化（这是分解求解的默认，本次不改）；而 readout 的默认
    打分是**原始点积**（不归一化），对应论文 readout probe 的语义。二者默认行为
    不同，是刻意保留的：分解关注几何重构，readout 关注 pre-softmax logit 排序。
    """
    target = target.detach().float().cpu()
    D_raw = dictionary.detach().float().cpu()
    D = F.normalize(D_raw, dim=1) if normalize_atoms else D_raw
    residual = target.clone()
    active: List[int] = []
    coeffs = torch.empty(0)

    for _ in range(min(k, D.shape[0])):
        scores = D @ residual
        if active:
            scores[torch.tensor(active, dtype=torch.long)] = -float("inf")
        best = int(torch.argmax(scores).item())
        best_score = float(scores[best].item())
        if best_score <= tol or not math.isfinite(best_score):
            break
        active.append(best)
        coeffs = nnls_pgd(D[active], target)
        reconstruction = coeffs @ D[active]
        residual = target - reconstruction
        if float(residual.norm().item()) <= tol:
            break

    if active:
        reconstruction = coeffs @ D[active]
        residual = target - reconstruction
    else:
        reconstruction = torch.zeros_like(target)
        residual = target.clone()
        coeffs = torch.empty(0)

    target_norm = float(target.norm().item())
    residual_norm = float(residual.norm().item())
    explained = 0.0 if target_norm <= EPS else 1.0 - (residual_norm * residual_norm) / (target_norm * target_norm)
    active_token_ids = [int(token_ids[i]) for i in active]
    result = DecompositionResult(
        active_indices=active,
        active_token_ids=active_token_ids,
        active_token_texts=[token_label(tokenizer, tid, token_texts) for tid in active_token_ids],
        coeffs=[float(x) for x in coeffs.tolist()],
        residual_norm=residual_norm,
        target_norm=target_norm,
        explained_fraction=float(explained),
    )
    return result, reconstruction, residual


def projection(h: Tensor, v: Tensor) -> Tensor:
    return ((h @ v) / (v @ v).clamp_min(EPS)) * v


def coordinate_patch(h: Tensor, v_source: Tensor, v_target: Tensor) -> Tensor:
    V = torch.stack([v_source, v_target], dim=1)  # [d, 2]
    coeffs = torch.linalg.pinv(V) @ h
    swapped = torch.flip(coeffs, dims=[0])
    return h + V @ (swapped - coeffs)


def load_dictionary(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"Unsupported dictionary format: {payload.get('format_version')}")
    return payload


def save_dictionary(path: str, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def layer_payload(payload: Mapping[str, Any], layer: int) -> Dict[str, Any]:
    layers = payload["layers"]
    key = str(layer)
    if key not in layers:
        available = ", ".join(sorted(layers.keys(), key=lambda x: int(x)))
        raise KeyError(f"Layer {layer} not in dictionary. Available: {available}")
    return layers[key]


def build_dictionary(args: argparse.Namespace) -> None:
    cfg = BuildConfig(
        model_id=args.model_id,
        layers=args.layers,
        max_length=args.max_length,
        max_prompts=args.max_prompts,
        max_pairs=args.max_pairs,
        position_mode=args.position_mode,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        normalize_saved_vectors=args.normalize_saved_vectors,
    )
    prompts = load_lines(args.prompts_file, DEFAULT_PROMPTS)
    if cfg.max_prompts is not None:
        prompts = prompts[: cfg.max_prompts]
    candidate_texts = load_lines(args.candidates_file, DEFAULT_CANDIDATES)

    model, tokenizer = load_model_and_tokenizer(
        cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        device_map=cfg.device_map,
        load_in_4bit=cfg.load_in_4bit,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
        dequantize_mxfp4=args.dequantize_mxfp4,
    )
    blocks, block_path = find_decoder_blocks(model)
    layers = parse_layers(cfg.layers, len(blocks))
    token_ids, token_to_texts = resolve_candidate_token_ids(tokenizer, candidate_texts)
    eprint(f"model={cfg.model_id} block_path={block_path} n_layers={len(blocks)}")
    eprint(f"layers={layers}")
    eprint(f"candidate_tokens={len(token_ids)} prompts={len(prompts)}")

    layer_dict: Dict[str, Any] = {}
    for layer_idx in layers:
        vectors, counts = estimate_token_vectors_for_layer(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            prompts=prompts,
            layer_idx=layer_idx,
            token_ids=token_ids,
            max_length=cfg.max_length,
            max_pairs=cfg.max_pairs,
            position_mode=cfg.position_mode,
        )
        if cfg.normalize_saved_vectors:
            vectors = F.normalize(vectors, dim=1)
        layer_dict[str(layer_idx)] = {
            "token_ids": [int(x) for x in token_ids],
            "vectors": vectors.contiguous(),
            "counts": counts,
            "norms": torch.linalg.norm(vectors, dim=1),
        }
        save_dictionary(
            args.out,
            {
                "format_version": FORMAT_VERSION,
                "model_id": cfg.model_id,
                "block_path": block_path,
                "config": asdict(cfg),
                "calibration_prompts": calibration_prompts_metadata(prompts),
                "candidate_texts": candidate_texts,
                "token_to_texts": {str(k): v for k, v in token_to_texts.items()},
                "layers": layer_dict,
            },
        )
        eprint(f"saved layer={layer_idx} to {args.out}")


def inspect_model(args: argparse.Namespace) -> None:
    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dequantize_mxfp4=args.dequantize_mxfp4,
    )
    blocks, block_path = find_decoder_blocks(model)
    info = {
        "model_id": args.model_id,
        "block_path": block_path,
        "num_layers": len(blocks),
        "vocab_size": getattr(model.config, "vocab_size", None),
        "hidden_size": getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd", None)
        or getattr(model.config, "d_model", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


def readout(args: argparse.Namespace) -> None:
    payload = load_dictionary(args.dictionary)
    model_id = args.model_id or payload.get("model_id", DEFAULT_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dequantize_mxfp4=args.dequantize_mxfp4,
    )
    blocks, _ = find_decoder_blocks(model)
    lp = layer_payload(payload, args.layer)
    token_ids = [int(x) for x in lp["token_ids"]]
    token_to_texts = {int(k): v for k, v in payload.get("token_to_texts", {}).items()}
    activation = capture_layer_activation(
        model, tokenizer, blocks, args.prompt, args.layer, args.position, args.max_length
    )
    rows = score_activation(
        activation,
        lp["vectors"],
        token_ids,
        token_to_texts=token_to_texts,
        tokenizer=tokenizer,
        top_k=args.top_k,
        cosine=args.cosine,
    )
    print(json.dumps({"prompt": args.prompt, "layer": args.layer, "position": args.position, "top": rows}, ensure_ascii=False, indent=2))


def readout_full(args: argparse.Namespace) -> None:
    verify_payload: Optional[Dict[str, Any]] = None
    verify_cfg: Mapping[str, Any] = {}
    if args.verify_against_dictionary:
        verify_payload = load_dictionary(args.verify_against_dictionary)
        verify_cfg = verify_payload.get("config", {}) or {}

    # When verifying against a saved dictionary, inherit its averaging settings unless
    # explicitly overridden, so the gate compares the same mean-Jacobian operator.
    max_length = int(args.max_length if args.max_length is not None else verify_cfg.get("max_length", 128))
    max_prompts = args.max_prompts if args.max_prompts is not None else verify_cfg.get("max_prompts")
    max_pairs = int(args.max_pairs if args.max_pairs is not None else verify_cfg.get("max_pairs", 1))
    position_mode = args.position_mode or str(verify_cfg.get("position_mode", "last"))

    prompts = load_lines(args.prompts_file, DEFAULT_PROMPTS)
    if max_prompts is not None:
        prompts = prompts[: int(max_prompts)]
    calibration_meta = calibration_prompts_metadata(prompts)
    if verify_payload is not None:
        # Fail before loading the model if the dictionary cannot prove calibration-set identity.
        calibration_meta = assert_same_calibration_prompts(verify_payload, prompts)

    model_id = args.model_id or (verify_payload or {}).get("model_id", DEFAULT_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dequantize_mxfp4=args.dequantize_mxfp4,
    )
    blocks, block_path = find_decoder_blocks(model)
    layer_spec = args.layers if args.layers else str(args.layer)
    if layer_spec in {"None", ""}:
        raise ValueError("readout-full requires --layer or --layers")
    layers = parse_layers(layer_spec, len(blocks))

    g_weight = find_final_norm_weight(model)
    if g_weight is None:
        eprint("[j-lens] 未找到 final-norm 可学习增益 g，readout-full 退化为 raw W_U。")
    tracked_token_ids = resolve_tracked_token_ids(tokenizer, args.track_token, args.track_token_id)

    layer_results: List[Dict[str, Any]] = []
    for layer_idx in layers:
        activation = capture_layer_activation(
            model, tokenizer, blocks, args.prompt, layer_idx, args.position, max_length
        )
        mean_jv, jvp_count = estimate_average_jvp_for_layer(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            prompts=prompts,
            layer_idx=layer_idx,
            tangent=activation,
            max_length=max_length,
            max_pairs=max_pairs,
            position_mode=position_mode,
            jvp_dtype=args.jvp_dtype,
        )
        # Fixed口径: scores = W_U @ (diag(g) · mean_q[J_q · h_ℓ(query)]).
        # The no-g vector is kept only for optional g-vs-raw diagnostics.
        j_lens_vector = apply_final_norm_gain_to_vector(mean_jv, g_weight)
        full_scores = score_full_vocab_from_vector(model, j_lens_vector, cosine=args.cosine)
        result: Dict[str, Any] = {
            "layer": int(layer_idx),
            "jvp_samples": int(jvp_count),
            "top": rows_from_full_scores(full_scores, tokenizer, args.top_k),
        }
        if tracked_token_ids:
            tracked = tracked_rows_from_scores(full_scores, tokenizer, tracked_token_ids)
            result["tracked"] = tracked
        if args.include_vanilla:
            vanilla_vector = apply_final_norm_gain_to_vector(activation, g_weight)
            vanilla_scores = score_full_vocab_from_vector(model, vanilla_vector, cosine=args.cosine)
            result["vanilla_top"] = rows_from_full_scores(vanilla_scores, tokenizer, args.top_k)
            if tracked_token_ids:
                vanilla_tracked = tracked_rows_from_scores(
                    vanilla_scores, tokenizer, tracked_token_ids, prefix="vanilla"
                )
                by_tid = {int(row["token_id"]): row for row in result.get("tracked", [])}
                for row in vanilla_tracked:
                    base = by_tid.setdefault(int(row["token_id"]), {"token_id": int(row["token_id"]), "text": row["text"]})
                    base.update({k: v for k, v in row.items() if k not in {"token_id", "text"}})
                result["tracked"] = list(by_tid.values())
        if args.compare_raw_g:
            raw_scores = score_full_vocab_from_vector(model, mean_jv, cosine=args.cosine)
            result["g_vs_raw_spearman_full_vocab"] = spearman_corr(full_scores, raw_scores)
        if verify_payload is not None:
            try:
                result["verify_against_dictionary"] = verify_full_scores_against_dictionary(
                    full_scores, activation, verify_payload, layer_idx, tokenizer
                )
            except RegressionGateError as exc:
                result["verify_against_dictionary"] = exc.result
                verify_lp = layer_payload(verify_payload, layer_idx)
                result["random_token_sanity"] = random_token_sanity_from_scores(
                    full_scores,
                    tokenizer,
                    [int(x) for x in verify_lp["token_ids"]],
                    n_random=args.random_sanity_tokens,
                    seed=args.random_sanity_seed,
                    top_k=min(20, args.top_k),
                )
                layer_results.append(result)
                payload = {
                    "prompt": args.prompt,
                    "position": int(args.position),
                    "model_id": model_id,
                    "block_path": block_path,
                    "score_mode": "cosine_post_j_space" if args.cosine else "dot",
                    "calibration": {
                        "prompts_file": args.prompts_file,
                        "num_prompts": len(prompts),
                        "prompts_sha256": calibration_meta["prompts_sha256"],
                        "position_mode": position_mode,
                        "max_pairs": max_pairs,
                        "max_length": max_length,
                    },
                    "gate_failed": True,
                    "failed_layer": int(layer_idx),
                    "failure_message": str(exc),
                    "layers": layer_results,
                }
                if args.compact_json:
                    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                else:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                raise
            verify_lp = layer_payload(verify_payload, layer_idx)
            result["random_token_sanity"] = random_token_sanity_from_scores(
                full_scores,
                tokenizer,
                [int(x) for x in verify_lp["token_ids"]],
                n_random=args.random_sanity_tokens,
                seed=args.random_sanity_seed,
                top_k=min(20, args.top_k),
            )
        layer_results.append(result)

    payload = {
        "prompt": args.prompt,
        "position": int(args.position),
        "model_id": model_id,
        "block_path": block_path,
        "score_mode": "cosine_post_j_space" if args.cosine else "dot",
        "calibration": {
            "prompts_file": args.prompts_file,
            "num_prompts": len(prompts),
            "prompts_sha256": calibration_meta["prompts_sha256"],
            "position_mode": position_mode,
            "max_pairs": max_pairs,
            "max_length": max_length,
        },
        "gate_failed": False,
        "layers": layer_results,
    }
    if args.compact_json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def decompose(args: argparse.Namespace) -> None:
    payload = load_dictionary(args.dictionary)
    model_id = args.model_id or payload.get("model_id", DEFAULT_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dequantize_mxfp4=args.dequantize_mxfp4,
    )
    blocks, _ = find_decoder_blocks(model)
    lp = layer_payload(payload, args.layer)
    token_ids = [int(x) for x in lp["token_ids"]]
    token_to_texts = {int(k): v for k, v in payload.get("token_to_texts", {}).items()}
    activation = capture_layer_activation(
        model, tokenizer, blocks, args.prompt, args.layer, args.position, args.max_length
    )
    result, _, _ = positive_sparse_pursuit(
        activation,
        lp["vectors"],
        token_ids,
        token_texts=token_to_texts,
        tokenizer=tokenizer,
        k=args.k,
        normalize_atoms=not args.raw_atoms,
    )
    print(json.dumps({"prompt": args.prompt, "layer": args.layer, "position": args.position, "j_space": asdict(result)}, ensure_ascii=False, indent=2))


def vector_for_token(
    payload: Mapping[str, Any],
    tokenizer: Any,
    layer: int,
    token: Optional[str],
    token_id: Optional[int],
) -> Tuple[int, Tensor]:
    lp = layer_payload(payload, layer)
    token_ids = [int(x) for x in lp["token_ids"]]
    if token_id is None:
        if token is None:
            raise ValueError("Provide --token or --token-id")
        ids = tokenizer.encode(token if token.startswith(" ") else " " + token, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Could not tokenize {token!r}")
        token_id = int(ids[-1])
    if int(token_id) not in token_ids:
        raise KeyError(f"token_id={token_id} is not in dictionary for layer {layer}")
    idx = token_ids.index(int(token_id))
    return int(token_id), lp["vectors"][idx].float().cpu()


def forward_logits_with_intervention(
    model: Any,
    blocks: Sequence[torch.nn.Module],
    batch: Mapping[str, Tensor],
    layer: int,
    position: int,
    vector: Tensor,
    mode: str,
    alpha: float,
    vector2: Optional[Tensor] = None,
) -> Any:
    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        hidden = first_hidden_from_block_output(output)
        pos = normalize_position(position, hidden.shape[1])
        modified = hidden.clone()
        v = vector.to(device=hidden.device, dtype=hidden.dtype)
        h = modified[0, pos]
        if mode == "steer":
            new_h = h + float(alpha) * v
        elif mode == "ablate":
            new_h = h - projection(h.float(), v.float()).to(dtype=hidden.dtype)
        elif mode == "patch":
            if vector2 is None:
                raise ValueError("patch mode requires vector2")
            v2 = vector2.to(device=hidden.device, dtype=hidden.dtype)
            new_h = coordinate_patch(h.float(), v.float(), v2.float()).to(dtype=hidden.dtype)
        else:
            raise ValueError("mode must be steer, ablate, or patch")
        modified[0, pos] = new_h
        return replace_hidden_in_block_output(output, modified)

    handle = blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            return model(**batch, use_cache=False, return_dict=True)
    finally:
        handle.remove()


def intervene(args: argparse.Namespace) -> None:
    payload = load_dictionary(args.dictionary)
    model_id = args.model_id or payload.get("model_id", DEFAULT_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dequantize_mxfp4=args.dequantize_mxfp4,
    )
    blocks, _ = find_decoder_blocks(model)
    token_id, vector = vector_for_token(payload, tokenizer, args.layer, args.token, args.token_id)
    vector2 = None
    token2_id = None
    if args.mode == "patch":
        token2_id, vector2 = vector_for_token(payload, tokenizer, args.layer, args.token2, args.token2_id)

    device = infer_input_device(model)
    input_ids = encode_prompt(tokenizer, args.prompt, args.max_length, device)["input_ids"]
    attention_mask = torch.ones_like(input_ids)

    for _ in range(args.steps):
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = forward_logits_with_intervention(
            model, blocks, batch, args.layer, args.position, vector, args.mode, args.alpha, vector2
        )
        next_id = int(torch.argmax(outputs.logits[0, -1]).item())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=input_ids.device)], dim=1)
        attention_mask = torch.ones_like(input_ids)

    result = {
        "prompt": args.prompt,
        "layer": args.layer,
        "position": args.position,
        "mode": args.mode,
        "token_id": token_id,
        "token_text": token_label(tokenizer, token_id),
        "token2_id": token2_id,
        "alpha": args.alpha,
        "generated": tokenizer.decode(input_ids[0], skip_special_tokens=True),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="传 'none' 可禁用 accelerate 的 device_map。")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--dequantize-mxfp4",
        default="auto",
        choices=["auto", "on", "off"],
        help="MXFP4（gpt-oss）反量化为 bf16：auto=探测到 mxfp4 config 即开启（默认，"
        "J-lens 求 VJP 及非 Blackwell GPU 必需）；on=强制；off=关闭。",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-OSS-20B 上的 J-Space / Jacobian Lens 复现")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect-model", help="加载模型并打印 decoder block 路径与层数等元信息。")
    add_model_args(p)
    p.set_defaults(func=inspect_model)

    p = sub.add_parser("build-dictionary", help="通过 VJP 平均估计 J-lens token 向量并存成字典。")
    add_model_args(p)
    p.add_argument("--prompts-file")
    p.add_argument("--candidates-file")
    p.add_argument("--layers", default="all", help="all、逗号列表、区间 a-b，或 Python 风格 a:b")
    p.add_argument("--out", default="jspace_dictionary.pt")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-prompts", type=int)
    p.add_argument("--max-pairs", type=int, default=1)
    p.add_argument("--position-mode", default="last", choices=["last", "all-same", "causal-window"])
    p.add_argument("--normalize-saved-vectors", action="store_true")
    p.set_defaults(func=build_dictionary)

    p = sub.add_parser("readout", help="J-lens readout：用字典对某个 activation 打分并排序。")
    add_model_args(p)
    p.add_argument("--dictionary", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--cosine", action="store_true", help="改用余弦相似度打分（默认使用论文口径的原始点积 ⟨v_t, h_ℓ⟩）。")
    p.set_defaults(func=readout)

    p = sub.add_parser("readout-full", help="全 vocab JVP readout：scores = W_U @ (diag(g)·mean J·h_l)；MoE double-backward patch 仅在 JVP 段临时生效并自动还原。")
    add_model_args(p)
    p.add_argument("--prompt", required=True, help="query prompt，用于捕获 h_l(query)。")
    p.add_argument("--layer", type=int, help="单层 readout；若提供 --layers 则可省略。")
    p.add_argument("--layers", help="all、逗号列表、区间 a-b，或 Python 风格 a:b；用于逐层轨迹。")
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--prompts-file", help="calibration prompts；默认使用内置 DEFAULT_PROMPTS。")
    p.add_argument("--position-mode", choices=["last", "all-same", "causal-window"], help="默认 last；验证字典时默认继承字典配置。")
    p.add_argument("--max-pairs", type=int, help="每个 calibration prompt 的位置对数；验证字典时默认继承字典配置。")
    p.add_argument("--max-prompts", type=int, help="截断 calibration prompts；验证字典时默认继承字典配置。")
    p.add_argument("--max-length", type=int, help="tokenization 最大长度；验证字典时默认继承字典配置，否则 128。")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--cosine", action="store_true", help="诊断用 post-J-space cosine；默认关，论文口径为点积。")
    p.add_argument("--jvp-dtype", default="auto", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"], help="诊断用：仅改变 double-backward JVP 的 y/u/tangent dtype，不提升模型权重/matmul；不得作为高精度参照。默认 auto 保持原行为。")
    p.add_argument("--verify-against-dictionary", help="强制回归门：把全 vocab 分数切到字典候选 token 并对拍现有 readout。")
    p.add_argument("--random-sanity-tokens", type=int, default=200, help="验证字典时额外报告的固定随机 token 数；仅做全 vocab sanity，不参与字典 Spearman 门。")
    p.add_argument("--random-sanity-seed", type=int, default=0, help="随机 token sanity 的固定种子。")
    p.add_argument("--track-token", action="append", help="额外报告该 token 的全 vocab rank；可重复。")
    p.add_argument("--track-token-id", action="append", type=int, help="额外报告该 token id 的全 vocab rank；可重复。")
    p.add_argument("--include-vanilla", action="store_true", help="同时输出 vanilla logit-lens（不过 J，仅 diag(g)·h_l）rank/top-k。")
    p.add_argument("--compare-raw-g", action="store_true", help="输出 full-vocab 折 g vs 不折 g 的 Spearman 诊断。")
    p.add_argument("--compact-json", action="store_true", help="单行 JSON，便于写 JSONL。")
    p.set_defaults(func=readout_full)

    p = sub.add_parser("decompose", help="对某个 activation 做稀疏非负 J-space 分解。")
    add_model_args(p)
    p.add_argument("--dictionary", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--raw-atoms", action="store_true", help="pursuit 前不对原子做归一化。")
    p.set_defaults(func=decompose)

    p = sub.add_parser("intervene", help="带 J-space steering/ablation/patching hook 的贪心生成。")
    add_model_args(p)
    p.add_argument("--dictionary", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--mode", choices=["steer", "ablate", "patch"], default="steer")
    p.add_argument("--token")
    p.add_argument("--token-id", type=int)
    p.add_argument("--token2")
    p.add_argument("--token2-id", type=int)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.set_defaults(func=intervene)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
