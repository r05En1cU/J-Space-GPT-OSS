# 在 GPT-OSS-20B 上复现 J-Space / Jacobian Lens

本目录实现了一个面向 HuggingFace decoder-only 模型（例如 GPT-OSS-20B）的 **Jacobian Lens** 与 **J-space** 实用复现路径，对应 `ALGORITHM.md` 中整理的算法描述。

论文来源（Paper Origin）：https://transformer-circuits.pub/2026/workspace/index.html

## 已实现内容

论文算法定义了从中间 residual stream 到最终 residual/logit space 的平均 Jacobian。对 20B 级模型而言，本实现采用等价且更可操作的 token-vector 形式：

```text
v_{layer, token} = E[ d logit_token(target_position) / d h_layer(source_position) ]
```

这就是 J-lens 所需的 token 方向，可用于：

- **J-lens readout**：将某层 activation 与可语言化 token 方向打分。
- **J-space decomposition**：在 token vectors 上做稀疏非负分解。
- **Steering**：向某个 residual stream 位置注入概念向量。
- **Ablation**：移除某个概念方向上的投影。
- **Coordinate patching**：在局部 J-space span 内交换两个概念坐标。

本实现刻意避免为每层物化完整的 `d_model x d_model` Jacobian。它通过 layer-output hook 直接计算 vector-Jacobian product，并在选定层截断计算图，因此早期 block 不需要保留梯度。

## 文件说明

- `ALGORITHM.md` — 概念算法说明。
- `jspace_gpt_oss.py` — 可运行实现与 CLI。
- `calibration_prompts.txt` — 默认校准 prompts。
- `candidate_concepts.txt` — 默认可语言化候选概念。
- `requirements.txt` — Python 依赖。
- `smoke_math_test.py` — 只测试数学辅助例程的轻量 smoke test。

## 安装

请使用适合目标模型的 GPU 环境。对 GPT-OSS-20B，建议在大显存 GPU 或多 GPU `device_map` 下使用 BF16。代码暴露了 4-bit 加载选项以支持资源受限的运行，但精确梯度行为取决于本地 `bitsandbytes` / `transformers` 栈。

```bash
pip install -r requirements.txt
```

如果模型已经缓存在本地，可以在命令中加入 `--local-files-only`。

## 检查模型结构

```bash
python jspace_gpt_oss.py inspect-model \
  --model-id openai/gpt-oss-20b \
  --torch-dtype bfloat16 \
  --device-map auto
```

该命令用于确认 decoder block 路径与层数。

## 构建 J-lens dictionary

建议先小规模跑通，再逐步放大。首次有意义的运行可以采样少量层、每个 prompt 一个 source/target pair，并使用默认候选概念列表：

```bash
python jspace_gpt_oss.py build-dictionary \
  --model-id openai/gpt-oss-20b \
  --prompts-file calibration_prompts.txt \
  --candidates-file candidate_concepts.txt \
  --layers 0,4,8,12,16,20 \
  --max-prompts 8 \
  --max-length 128 \
  --max-pairs 1 \
  --position-mode last \
  --torch-dtype bfloat16 \
  --device-map auto \
  --out gpt_oss_20b_jspace_dictionary.pt
```

若要更完整复现，可逐步增加：

- `--layers all`
- prompt 数量
- candidate concept 数量
- `--max-pairs`，并配合 `--position-mode causal-window` 或 `all-same`

## J-lens readout

```bash
python jspace_gpt_oss.py readout \
  --dictionary gpt_oss_20b_jspace_dictionary.pt \
  --model-id openai/gpt-oss-20b \
  --prompt "A spider builds a" \
  --layer 12 \
  --position -1 \
  --top-k 20
```

输出为 JSON，包含按分数排序的 token concepts。

## J-space decomposition

```bash
python jspace_gpt_oss.py decompose \
  --dictionary gpt_oss_20b_jspace_dictionary.pt \
  --model-id openai/gpt-oss-20b \
  --prompt "A spider builds a" \
  --layer 12 \
  --position -1 \
  --k 25
```

结果会报告激活的 token IDs/texts、非负系数、残差范数与解释比例。

## 因果干预

向某个概念 steering：

```bash
python jspace_gpt_oss.py intervene \
  --dictionary gpt_oss_20b_jspace_dictionary.pt \
  --model-id openai/gpt-oss-20b \
  --prompt "The animal crawled across the" \
  --layer 12 \
  --position -1 \
  --mode steer \
  --token " spider" \
  --alpha 2.0 \
  --steps 32
```

ablate 某个概念：

```bash
python jspace_gpt_oss.py intervene \
  --dictionary gpt_oss_20b_jspace_dictionary.pt \
  --model-id openai/gpt-oss-20b \
  --prompt "The animal crawled across the" \
  --layer 12 \
  --position -1 \
  --mode ablate \
  --token " spider" \
  --steps 32
```

patch 两个概念坐标：

```bash
python jspace_gpt_oss.py intervene \
  --dictionary gpt_oss_20b_jspace_dictionary.pt \
  --model-id openai/gpt-oss-20b \
  --prompt "The city is famous for the Eiffel Tower in" \
  --layer 12 \
  --position -1 \
  --mode patch \
  --token " Paris" \
  --token2 " London" \
  --steps 16
```

## 复现实验协议

建议记录下表：

| 阶段 | 目标 | 最小设置 | 放大设置 |
|---|---|---:|---:|
| Dictionary | 估计 J-lens token vectors | 6 层 × 8 prompts × 40 concepts | 全层 × 100+ prompts × 1k+ concepts |
| Readout | 验证可语言化概念 | top-20 cosine scores | 与 logit lens baseline 对比 |
| Decomposition | 估计 J-space 稀疏性 | `k=25` | sweep `k ∈ {5, 10, 25, 50}` |
| Intervention | 验证因果效应 | steer/ablate selected concepts | paired prompts + effect-size table |

核心指标：

- readout top-k 与人类预期概念的重合度；
- decomposition explained fraction 随 `k` 的变化；
- steering 对目标概念 logits 的增量；
- ablation 对目标概念 logits 的下降；
- patching 在成对 source/target prompts 上的成功率。

## 注意事项

- 本实现估计的是基于 VJP 的 **token-vector J-space**，不是存储完整平均 Jacobian 矩阵。
- candidate vocabulary size 会线性影响运行时间；建议先使用聚焦的概念列表，再扩展。
- 多 token candidate string 使用其最终 token ID 表示；这符合 token-level 定义，但分析时需要明确记录。
- 为保证正确性和实现简洁，greedy intervention generation 会禁用 KV cache，因此速度较慢。
- 如果本地环境中的模型 ID 不是 `openai/gpt-oss-20b`，请通过 `--model-id` 传入正确的 HuggingFace 路径。
