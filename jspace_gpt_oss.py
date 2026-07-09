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
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
