#!/usr/bin/env python3
"""真机校验脚本：确认 h_final 是否 pre-norm 残差，并量化 final-norm 逐通道增益 g
对 J-lens readout 排序的影响。

背景：J-lens token 向量当前严格取为 rows of raw `W_U J_ℓ`（final-norm 的可学习
逐通道增益 g 未折入）。论文口径称 ⟨v_t,h_ℓ⟩「up to a data-dependent normalization
factor」等于 pre-softmax logit——该因子应为标量（1/rms），但 g 是逐通道 diag(g)，
不是标量。若省略 g 会系统性扭曲 token 间排序，则应把 g 折入有效解嵌。

本脚本不修改主实现，只做诊断，回答两个问题：

  [Q3] blocks[-1] 的输出 tuple[0] 是否为 final-norm 之前的残差流？
       判据：logits[0,pos] ≈ W_U @ final_norm(h_final[0,pos])。

  [Q5] 折入 g 与否，readout 的 top-k 排序偏差有多大？
       比较 score_raw = ⟨W_U[token]·J_ℓ,     h_ℓ⟩
       与    score_g   = ⟨W_U[token]·diag(g)·J_ℓ, h_ℓ⟩
       用 Spearman 相关 + top-10 重合率衡量；偏差小→维持 raw，偏差大→折入 g。

用法（在有 GPU 的 StarPhoton 环境）：
    conda run -n StarPhoton python check_gnorm_alignment.py \
        --model-id openai/gpt-oss-20b --layer 12 --torch-dtype bfloat16 --device-map auto
若模型已本地缓存，可加 --local-files-only。
"""

from __future__ import annotations

import argparse
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from jspace_gpt_oss import (
    DEFAULT_CANDIDATES,
    DEFAULT_PROMPTS,
    encode_prompt,
    find_decoder_blocks,
    first_hidden_from_block_output,
    infer_input_device,
    load_model_and_tokenizer,
    normalize_position,
    resolve_candidate_token_ids,
)


def find_final_norm(model: Any) -> Optional[torch.nn.Module]:
    """定位 unembedding 之前的最终 norm 层（final-norm）。

    覆盖主流 HF decoder 命名：model.norm / model.final_layernorm /
    transformer.ln_f / gpt_neox.final_layer_norm 等。找不到则返回 None。
    """
    candidates = [
        "model.norm",
        "model.final_layernorm",
        "model.final_layer_norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
        "model.decoder.final_layer_norm",
    ]
    for path in candidates:
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    return None


def get_norm_weight(norm_module: Optional[torch.nn.Module]) -> Optional[torch.Tensor]:
    """取 final-norm 的逐通道可学习增益 g（RMSNorm/LayerNorm 的 weight）。"""
    if norm_module is None:
        return None
    w = getattr(norm_module, "weight", None)
    return w.detach() if w is not None else None


def capture_leaf_and_final(
    model: Any, blocks, layer_idx: int, batch
) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """一次前向，返回 (outputs, leaf@layer_idx, h_final@blocks[-1])，图经 leaf 相连。"""
    captured = {}

    def leaf_hook(_m, _i, output):
        hidden = first_hidden_from_block_output(output)
        leaf = hidden.detach().requires_grad_(True)
        captured["leaf"] = leaf
        from jspace_gpt_oss import replace_hidden_in_block_output

        return replace_hidden_in_block_output(output, leaf)

    def final_hook(_m, _i, output):
        captured["h_final"] = first_hidden_from_block_output(output)
        return None

    handles = [
        blocks[layer_idx].register_forward_hook(leaf_hook),
        blocks[-1].register_forward_hook(final_hook),
    ]
    try:
        with torch.enable_grad():
            outputs = model(**batch, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return outputs, captured["leaf"], captured["h_final"]


def main() -> None:
    ap = argparse.ArgumentParser(description="J-lens h_final/g 对齐诊断")
    ap.add_argument("--model-id", default="openai/gpt-oss-20b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--position", type=int, default=-1)
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--dequantize-mxfp4", default="auto", choices=["auto", "on", "off"],
                    help="MXFP4 反量化为 bf16（auto=探测到即开启，VJP 求梯度必需）。")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=4, help="用前 N 条默认 prompt 估计 J_ℓ 平均")
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        dequantize_mxfp4=args.dequantize_mxfp4,
    )
    blocks, block_path = find_decoder_blocks(model)
    print(f"[info] block_path={block_path} n_layers={len(blocks)} layer={args.layer}")

    W_U = model.get_output_embeddings().weight.detach()
    final_norm = find_final_norm(model)
    g = get_norm_weight(final_norm)
    print(f"[info] final_norm={type(final_norm).__name__ if final_norm else None} "
          f"g_weight={'found shape=' + str(tuple(g.shape)) if g is not None else 'NOT FOUND'}")

    device = infer_input_device(model)

    # ---------- [Q3] h_final 是否 pre-norm：单 prompt logit 重构校验 ----------
    prompt0 = DEFAULT_PROMPTS[0]
    batch0 = encode_prompt(tokenizer, prompt0, args.max_length, device)
    outputs0, _leaf0, h_final0 = capture_leaf_and_final(model, blocks, args.layer, batch0)
    seq = h_final0.shape[1]
    pos = normalize_position(args.position, seq)
    h_last = h_final0[0, pos].detach().float()

    if final_norm is not None:
        with torch.no_grad():
            recon = final_norm(h_final0[0, pos].unsqueeze(0)).squeeze(0).float()
        recon_logits = (W_U.float().to(recon.device) @ recon.to(W_U.device).float())
        true_logits = outputs0.logits[0, pos].detach().float().to(recon_logits.device)
        # 比较分布（去均值后 cosine + top-10 重合），logit 有整体偏移不影响 argmax
        c = F.cosine_similarity(
            (recon_logits - recon_logits.mean()).unsqueeze(0),
            (true_logits - true_logits.mean()).unsqueeze(0),
        ).item()
        t_true = set(torch.topk(true_logits, 10).indices.tolist())
        t_recon = set(torch.topk(recon_logits, 10).indices.tolist())
        overlap = len(t_true & t_recon)
        print("\n===== [Q3] h_final 是否 pre-norm 残差 =====")
        print(f"  cosine(centered recon_logits, true_logits) = {c:.4f}  (期望≈>0.99)")
        print(f"  top-10 argmax 重合 = {overlap}/10  (期望=10 或近 10)")
        print(f"  判定: {'✅ h_final 确为 pre-norm 残差' if c > 0.95 and overlap >= 8 else '⚠️ 不吻合，h_final 可能已含 norm 或 block 路径异常'}")
    else:
        print("\n[Q3] 未定位 final_norm，跳过 logit 重构；请手动确认 unembed 前的 norm 层名。")

    # ---------- [Q5] 折入 g 与否的排序偏差 ----------
    # 估计该层 J-lens 向量：raw = W_U[t]·J_ℓ, g版 = W_U[t]·diag(g)·J_ℓ
    cand_ids, _ = resolve_candidate_token_ids(tokenizer, DEFAULT_CANDIDATES)
    cand_ids = cand_ids[:40]
    prompts = DEFAULT_PROMPTS[: args.n_prompts]

    # 在同一 activation 上打分：用最后一条 prompt 的 layer activation 作为 h_ℓ
    # （诊断目的，取单点即可）
    sums_raw = None
    sums_g = None
    count = 0
    g_vec = g.float() if g is not None else None
    for p in prompts:
        batch = encode_prompt(tokenizer, p, args.max_length, device)
        outputs, leaf, h_final = capture_leaf_and_final(model, blocks, args.layer, batch)
        tp = normalize_position(args.position, h_final.shape[1])
        sp = normalize_position(args.position, leaf.shape[1])
        h_vec = h_final[0, tp]
        if g_vec is not None:
            g_on = g_vec.to(h_vec.device, h_vec.dtype)
        d_model = leaf.shape[-1]
        if sums_raw is None:
            sums_raw = torch.zeros((len(cand_ids), d_model), dtype=torch.float32)
            sums_g = torch.zeros((len(cand_ids), d_model), dtype=torch.float32)
        for i, tid in enumerate(cand_ids):
            w = W_U[int(tid)].to(h_vec.device, h_vec.dtype)
            s_raw = (w * h_vec).sum()
            grad_raw = torch.autograd.grad(s_raw, leaf, retain_graph=True)[0]
            sums_raw[i] += grad_raw[0, sp].detach().float().cpu()
            if g_vec is not None:
                s_g = (w * g_on * h_vec).sum()
                grad_g = torch.autograd.grad(s_g, leaf, retain_graph=True)[0]
                sums_g[i] += grad_g[0, sp].detach().float().cpu()
        count += 1
        del outputs, leaf, h_final
    V_raw = sums_raw / count
    V_g = sums_g / count if g_vec is not None else None

    # 用一条探测 activation（第一个 prompt 的该层 block 输出）做 readout 打分对比
    probe = {}

    def probe_hook(_m, _i, output):
        probe["h"] = first_hidden_from_block_output(output)[0].detach().float().cpu()
        return output

    hb = blocks[args.layer].register_forward_hook(probe_hook)
    try:
        with torch.no_grad():
            model(**encode_prompt(tokenizer, prompt0, args.max_length, device),
                  use_cache=False, return_dict=True)
    finally:
        hb.remove()
    h_probe = probe["h"][pos]

    score_raw = V_raw @ h_probe
    print("\n===== [Q5] 折入 g 与否的 readout 排序偏差 =====")
    if V_g is not None:
        score_g = V_g @ h_probe

        def spearman(a, b):
            ra = a.argsort().argsort().float()
            rb = b.argsort().argsort().float()
            ra = (ra - ra.mean()) / (ra.std() + 1e-8)
            rb = (rb - rb.mean()) / (rb.std() + 1e-8)
            return (ra * rb).mean().item()

        rho = spearman(score_raw, score_g)
        k = min(10, len(cand_ids))
        top_raw = set(torch.topk(score_raw, k).indices.tolist())
        top_g = set(torch.topk(score_g, k).indices.tolist())
        ov = len(top_raw & top_g)
        print(f"  Spearman(score_raw, score_g) = {rho:.4f}")
        print(f"  top-{k} 重合 = {ov}/{k}")
        top1_raw = int(score_raw.argmax()); top1_g = int(score_g.argmax())
        print(f"  raw top-1 token = {tokenizer.decode([cand_ids[top1_raw]])!r}  "
              f"g top-1 token = {tokenizer.decode([cand_ids[top1_g]])!r}")
        print("  判定:")
        if rho > 0.98 and ov >= k - 1:
            print("    ✅ 偏差极小 → 维持 raw（不折 g）在排序层面无实质影响")
        elif rho > 0.9:
            print("    🔶 中等偏差 → 建议折入 g 以复现真 logit 排序（尤其若 Q3 已确认 pre-norm）")
        else:
            print("    ⚠️ 显著偏差 → 应折入 g，否则 readout 排序系统性偏离模型真实 logit")
    else:
        print("  未找到 g（final_norm 无 weight，或为无仿射的 norm）→ 无需折入，raw 即正确。")


if __name__ == "__main__":
    main()
