# Plan B — 全 vocab JVP readout + 逐层浮现轨迹图 + Spearman 复现

## 动机
当前 `readout` 只在 ~40 个候选概念(dictionary)内排序(`score_activation`: `scores = D @ h`,
D = 候选向量)。答案 token 被预先塞进候选集再看它"浮现",判别力虚高,**不能对标论文
logit-lens 式的全 vocab 浮现图**。本计划补一条全 vocab readout 路径,产出论文招牌图。

## 核心恒等式(全 vocab readout 为何便宜)
readout 分数:
    scores[t] = ⟨v_t, h_ℓ⟩ = W_U[t]^T · diag(g) · J_ℓ · h_ℓ
令
    w = diag(g) · (J_ℓ · h_ℓ)          # d_model 向量,与 token 无关
则
    scores = W_U @ w                     # 全 vocab 一次 matmul
所以**不需要逐 token VJP**(那是建字典用的),单 prompt 出图只需算一次 `J_ℓ · h_ℓ`。

## 口径决定(reviewer 必查,不得擅改)
1. **J_ℓ 用平均 Jacobian**,与现有 `build-dictionary` 完全一致,不是 local-J:
       w = diag(g) ⊙ mean_q [ J_q · h_ℓ ]
   q 遍历 calibration prompts(`--prompts-file` / `DEFAULT_PROMPTS`),position 对齐现有
   `make_position_pairs`(默认 last→last)。`J_q · h_ℓ` = 在 prompt q 的 source_pos 注入
   tangent = query 的 h_ℓ,读 h_final 在 target_pos 的方向导数(JVP)。JVP 线性 ⇒
   mean_q[J_q·h_ℓ] = (mean_q J_q)·h_ℓ,即平均算子作用于 query 向量,语义正确。
2. **diag(g) 折入**(2026-07-09 定案),g = final-norm 逐通道增益;h_final 取 pre-norm。
3. readout **点积口径**(非 cosine)。
4. query 的 h_ℓ 用现有 `capture_layer_activation` 抓(detach 的 raw 激活)。

## JVP 实现(double-backward,兼容 HF hook)
PyTorch 前向模式对大模型 hook 兼容差,用 double-vjp 技巧算 `J·v`(J=∂h_final/∂leaf,v=tangent):
```
# 复用 forward_with_layer_leaf(model, blocks, ℓ, batch_q) 拿到 (leaf, h_final)
y = h_final[0, target_pos]                 # d 向量,连着 leaf
u = torch.zeros_like(y, requires_grad=True) # dummy
(Jt_u,) = torch.autograd.grad(y, leaf, grad_outputs=u, create_graph=True)  # J^T u,关于 u 可导
(Jv,)   = torch.autograd.grad(Jt_u, u, grad_outputs=v_at_source)           # = J·v
# v_at_source: 与 leaf 同形的零张量,仅 [0, source_pos] = query_h_ℓ(对齐 device/dtype)
```
每个 calibration prompt 2 次 backward;N_cal 个 prompt 求平均得 `mean_q[J_q·h_ℓ]`。
**bf16 注入口径**:当前实现会把 float32 tangent round 到 leaf 的 bf16 dtype 后注入(见 `v_at_source[0, source_pos] = ... dtype=leaf.dtype`),因此整个二阶图主要在 bf16 中运行；若某层 rho 卡在 0.999 边缘,先把 y/u 或注入 tangent 提 fp32 再试。
**跨设备**:device_map=auto 下 leaf / h_final / W_U / g 可能不同 device,逐处 `.to()` 对齐
(照抄现有 `estimate_token_vectors_for_layer` 的对齐写法)。

## 新增子命令 `readout-full`
参数:`--prompt`(query)、`--layer`、`--position`(默认 -1)、`--prompts-file`(calibration)、
`--position-mode`(默认 last)、`--max-pairs`、`--top-k`、`--layers`(可选,扫多层出轨迹)、
`--cosine`(默认关)。输出 JSON:每层 topk 的 {rank, token_id, text, score}。
不依赖已存 dictionary(现算),但 `--model-id` / `--dequantize-mxfp4` 等复用 `add_model_args`。

## 强制回归门(reviewer 验收硬标准)
把 `readout-full` 的全 vocab 分数**切片到现有 40 个候选 token**,其排序必须与现有
`readout`(dictionary 点积)**一致**:Spearman ≈ 1.0、top-10 同序。
> 不一致 ⇒ 平均 Jacobian 的 JVP 口径写错(方向/转置/position/平均搞反),**直接打回**。
> 这是口径正确性的经验判据,优先级高于任何解析论证。
实现一个 `--verify-against-dictionary <dict.pt>` 开关跑这个对拍,打印 Spearman 与 top-10 交集。

## Spearman 0.705 复现(口径回归)
`readout-full` 上线后,复用 gcheck 思路在**全 vocab**上重跑"折 g vs 不折 g"的 readout 排序
Spearman。记录全 vocab 版新数(可能≠候选集的 0.705,如实记),确认折 g 仍显著改善排序。

## 出图(交付图)
1. **浮现轨迹图**:x=layer(0,4,8,12,16,20),y=答案 token 在全 vocab 的 rank(log 轴),
   多条探针(France→Paris、spider→ web/insect… 用记忆里那批)叠加。
2. **招牌对比图**:同一探针,**J-Lens**(过 J) vs **vanilla logit-lens**(w=diag(g)·h_ℓ 直接
   过 W_U,不过 J)两条 rank 曲线,展示 J-Lens 答案更早/更干净浮现。
3. 探针清单先照现有 `calibration_prompts.txt`,验证管线通后再扩到论文规模(注释已写 100+)。

## 分工
- 实现:grunt-worker 按本计划(核心数学已拍死,执行为主)。
- 复核:**opus-code-reviewer 必审** JVP 口径 / 回归门 / 跨设备 / 末层退化。
- 若 reviewer 对"平均-vs-local Jacobian"或 JVP 转置方向存疑 → 升级 `fable-advisor [思考档:max]`。
- 上机:走 `gptoss` conda env,A100,`--local-files-only`,MXFP4 auto 反量化(照 run_gpt_oss_a100.sh)。
