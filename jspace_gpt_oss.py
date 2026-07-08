#!/usr/bin/env python3
"""
Practical J-Space / Jacobian Lens reproduction for GPT-OSS-20B-style
HuggingFace decoder-only language models.

The paper algorithm materializes an average Jacobian J_l from an intermediate
residual stream h_l to the final residual/logit readout. For a 20B model, the
most useful reproducible object is the equivalent token-vector dictionary:

    v_{l, token} = E[ d logit_token(t') / d h_l(t) ]

This is J_l^T times the unembedding/norm readout, evaluated by VJP. It avoids
building full d_model x d_model Jacobians while preserving the J-space objects
needed for readout, sparse nonnegative decomposition, steering, ablation, and
coordinate patching.
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
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: transformers. Install with `pip install transformers accelerate`."
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
    position_mode: str = "last"  # last | all-same | causal-window
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
        raise ValueError(f"Unsupported dtype {name!r}; use auto/fp32/fp16/bf16.")
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

    # Conservative fallback: choose the largest ModuleList with transformer-like blocks.
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
        "Could not locate decoder blocks. Add the model's block path to COMMON_BLOCK_PATHS."
    )


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
            # J-space is token-level. Multi-token strings contribute their final token,
            # which is usually the semantic carrier for GPT-style BPE/SentencePiece.
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


def load_model_and_tokenizer(
    model_id: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
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
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise SystemExit("--load-in-4bit requires bitsandbytes and a recent transformers build.") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # We only need gradients with respect to the hooked residual stream leaf.
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
) -> Tuple[Any, Tensor]:
    captured: Dict[str, Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        hidden = first_hidden_from_block_output(output)
        leaf = hidden.detach().requires_grad_(True)
        captured["leaf"] = leaf
        return replace_hidden_in_block_output(output, leaf)

    handle = blocks[layer_idx].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            outputs = model(**batch, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if "leaf" not in captured:
        raise RuntimeError(f"Layer hook did not capture layer {layer_idx}")
    return outputs, captured["leaf"]


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
    if not token_ids:
        raise ValueError("No candidate token ids provided.")
    device = infer_input_device(model)
    d_model: Optional[int] = None
    sums: Optional[Tensor] = None
    counts = torch.zeros(len(token_ids), dtype=torch.long)

    for prompt_idx, prompt in enumerate(prompts):
        batch = encode_prompt(tokenizer, prompt, max_length, device)
        seq_len = int(batch["input_ids"].shape[1])
        pairs = make_position_pairs(seq_len, position_mode, max_pairs)
        for pair_idx, (source_pos, target_pos) in enumerate(pairs):
            outputs, leaf = forward_with_layer_leaf(model, blocks, layer_idx, batch)
            if d_model is None:
                d_model = int(leaf.shape[-1])
                sums = torch.zeros((len(token_ids), d_model), dtype=torch.float32)
            source_pos = normalize_position(source_pos, leaf.shape[1])
            target_pos = normalize_position(target_pos, outputs.logits.shape[1])
            logits = outputs.logits[0, target_pos]
            for i, token_id in enumerate(token_ids):
                logit = logits[int(token_id)].float()
                grad = torch.autograd.grad(
                    logit,
                    leaf,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                vec = grad[0, source_pos].detach().float().cpu()
                assert sums is not None
                sums[i].add_(vec)
                counts[i] += 1
            del outputs, leaf, logits
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
    cosine: bool = True,
) -> List[Dict[str, Any]]:
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
    """Small projected-gradient NNLS solve for active atoms.

    D_active: [m, d], target: [d]. Returns coeffs [m] >= 0.
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
        cosine=not args.dot_product,
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
    parser.add_argument("--device-map", default="auto", help="Use 'none' to disable accelerate device_map.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="J-Space / Jacobian Lens reproduction for GPT-OSS-20B")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect-model", help="Load a model and print block/layer metadata.")
    add_model_args(p)
    p.set_defaults(func=inspect_model)

    p = sub.add_parser("build-dictionary", help="Estimate J-lens token vectors by VJP averaging.")
    add_model_args(p)
    p.add_argument("--prompts-file")
    p.add_argument("--candidates-file")
    p.add_argument("--layers", default="all", help="all, comma list, range a-b, or Python-style a:b")
    p.add_argument("--out", default="jspace_dictionary.pt")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-prompts", type=int)
    p.add_argument("--max-pairs", type=int, default=1)
    p.add_argument("--position-mode", default="last", choices=["last", "all-same", "causal-window"])
    p.add_argument("--normalize-saved-vectors", action="store_true")
    p.set_defaults(func=build_dictionary)

    p = sub.add_parser("readout", help="J-lens readout: score one activation against the dictionary.")
    add_model_args(p)
    p.add_argument("--dictionary", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--dot-product", action="store_true", help="Use raw dot product instead of cosine scores.")
    p.set_defaults(func=readout)

    p = sub.add_parser("decompose", help="Sparse nonnegative J-space decomposition for one activation.")
    add_model_args(p)
    p.add_argument("--dictionary", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--raw-atoms", action="store_true", help="Do not normalize atoms before pursuit.")
    p.set_defaults(func=decompose)

    p = sub.add_parser("intervene", help="Greedy generation with J-space steering/ablation/patching hook.")
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
