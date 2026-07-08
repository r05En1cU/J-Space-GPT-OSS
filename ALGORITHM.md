# J-Space / Jacobian Lens 核心算法说明

本文档整理自 Transformer Circuits 论文 **Verbalizable Representations Form a Global Workspace in Language Models**，目标是用常规 **decoder-only Transformer** 架构解释论文的核心算法：**Jacobian Lens（J-lens）** 与 **J-space**。

> 一句话概括：**Jacobian Lens 用平均 Jacobian 把中间层 residual stream 映射到最终可说出的 token 概念；J-space 是这些 token 概念方向组成的稀疏“可语言化工作区”。**

---

## 1. 常规 decoder-only Transformer 背景

标准 decoder-only Transformer 的计算路径可以简化为：

```text
tokens
  ↓
token embedding + position embedding
  ↓
residual stream h_0
  ↓
[ Transformer Block 1 ]
  ↓
residual stream h_1
  ↓
...
  ↓
[ Transformer Block L ]
  ↓
final residual stream h_L
  ↓
final norm
  ↓
unembedding W_U
  ↓
logits over vocabulary
  ↓
softmax → next-token distribution
```

记号约定：

- `h_{ℓ,t}`：第 `ℓ` 层、第 `t` 个 token 位置的 residual stream 向量。
- `L`：Transformer 总层数。
- `W_U`：unembedding matrix，把最终 residual 表示映射成 vocabulary logits。
- decoder-only Transformer 是 causal 的，因此位置 `t` 的表示只能影响当前位置或未来位置 `t' >= t`。

普通 **logit lens** 直接把中间层 `h_ℓ` 接到 `W_U` 上：

```text
logits_ℓ = W_U · norm(h_ℓ)
```

但这隐含假设：中间层 residual space 与最终层 residual space 已经自然对齐。

**Jacobian Lens** 的核心改进是：不直接读 `h_ℓ`，而是先估计它对最终层 `h_L` 的平均一阶影响。

---

## 2. Jacobian Lens 的目标

给定某一层 activation：

```text
h_{ℓ,t}
```

Jacobian Lens 想回答的问题不是：

```text
这一层直接预测下一个 token 是什么？
```

而是：

```text
如果这个中间表示影响最终层输出，
它平均会推动模型 verbalize 哪些 token / concept？
```

也就是说，J-lens 试图读出中间层中**可被语言化的内部概念**。

---

## 3. 平均 Jacobian：从中间层到最终层

考虑从中间层 residual stream 到最终层 residual stream 的局部 Jacobian：

```text
∂h_{L,t'} / ∂h_{ℓ,t}
```

其中：

- `ℓ` 是源层。
- `t` 是源 token 位置。
- `t' >= t` 是当前或未来 token 位置。
- `h_{L,t'}` 是最终层 residual stream。

论文对大量 prompt、source position 和 target position 求平均，得到每层一个平均 Jacobian：

```text
J_ℓ = E_{prompt, t, t' >= t} [ ∂h_{L,t'} / ∂h_{ℓ,t} ]
```

`J_ℓ` 可以理解为：

```text
第 ℓ 层 residual direction → 最终层 residual direction
```

的平均线性映射。

---

## 4. J-lens 读出公式

有了 `J_ℓ` 后，对任意中间层 activation `h_{ℓ,t}`，J-lens 读出为：

```text
h_final_approx = J_ℓ h_{ℓ,t}
logits = W_U · norm(h_final_approx)
```

也就是：

```text
J_lens(h_{ℓ,t}) = W_U · norm(J_ℓ h_{ℓ,t})
```

实际使用时，对 logits 排序：

```text
top_tokens = top_k(W_U · norm(J_ℓ h_{ℓ,t}))
```

这些 top tokens 就是该 activation 中最容易被模型 verbalize 的概念。

---

## 5. Jacobian Lens vs Logit Lens

### Logit Lens

```text
logits_ℓ = W_U · norm(h_ℓ)
```

特点：

- 简单直接。
- 假设中间层和最终层表示空间天然对齐。
- 对较早层或中间层可能不稳定。

### Jacobian Lens

```text
logits_ℓ = W_U · norm(J_ℓ h_ℓ)
```

特点：

- 显式估计从第 `ℓ` 层到最终层的平均线性变换。
- 更接近“一阶因果影响”：中间层某个方向如何影响最终输出。
- 更适合读取中间层中可语言化、可报告的概念。

---

## 6. J-lens token vector

对于每一层 `ℓ`，每个 vocabulary token 都可以对应一个 J-lens 方向。

如果读出 logits 为：

```text
logits = W_U · norm(J_ℓ h)
```

忽略 norm 的局部复杂性，可以把每个 token 的读出方向理解为：

```text
v_{ℓ, token} ≈ J_ℓ^T W_U[token]^T
```

直观上：

```text
v_{ℓ, "spider"}
```

表示在第 `ℓ` 层 residual stream 中，哪些方向会平均推动模型未来 verbalize `spider` 这个 token / concept。

所有 token 的 J-lens vectors 构成一个 overcomplete dictionary：

```text
D_ℓ = { v_{ℓ,1}, v_{ℓ,2}, ..., v_{ℓ,|V|} }
```

其中 `|V|` 是词表大小，通常远大于 `d_model`。

---

## 7. J-space：可语言化表示子空间

论文把由少量 J-lens token vectors 组成的稀疏非负组合称为 **J-space**。

形式化地，对 activation `h`，其 J-space component 可以写成：

```text
h_J = Σ_i a_i v_i
```

约束为：

```text
a_i >= 0
number of active i <= k
```

其中：

- `v_i` 是某个 token 的 J-lens vector。
- `a_i` 是该 concept 的激活强度。
- `k` 是稀疏度上限，论文中常用不超过约 25 个活跃方向。

因此一个 activation 可以被拆成：

```text
h = h_J + h_nonJ
```

含义是：

- `h_J`：可语言化、可被 J-lens 读出的部分。
- `h_nonJ`：其余不可直接 verbalize 的模型内部表示。

---

## 8. J-space 稀疏分解目标

给定 activation `h` 和 J-lens dictionary `D_ℓ`，求：

```text
minimize || h - Σ_i a_i v_i ||²

subject to:
  a_i >= 0
  ||a||_0 <= k
```

这就是一个 sparse nonnegative decomposition 问题。

论文使用 **gradient pursuit** 近似求解。输出结果是：

```text
active_vectors = [v_1, v_2, ..., v_m]
coefficients = [a_1, a_2, ..., a_m]
```

这些 active vectors 对应的 token 就是当前 activation 中活跃的 verbalizable concepts。

---

## 9. 核心算法流程

### Step 1：准备校准语料

准备一批 calibration prompts：

```text
prompt_1, prompt_2, ..., prompt_N
```

这些 prompts 用来估计平均 Jacobian。

---

### Step 2：hook residual stream

对每个 prompt 运行模型，并缓存每一层 residual stream：

```text
h_{0,t}, h_{1,t}, ..., h_{L,t}
```

需要 hook 的通常是每个 Transformer block 后的 residual stream。

---

### Step 3：估计每层平均 Jacobian

对每一层 `ℓ`：

```text
for prompt in prompts:
    run model forward
    cache residual streams

    for source position t:
        for target position t' >= t:
            compute J_sample = ∂h_{L,t'} / ∂h_{ℓ,t}
            collect J_sample

J_ℓ = mean(J_sample over prompts and positions)
```

得到每层一个矩阵：

```text
J_1, J_2, ..., J_L
```

---

### Step 4：构造 J-lens readout

对某层 activation `h_{ℓ,t}`：

```text
h_final_approx = J_ℓ h_{ℓ,t}
logits = W_U · norm(h_final_approx)
top_tokens = top_k(logits)
```

输出示例：

```text
Layer 32, position t:
  spider
  web
  legs
  animal
  insect
```

这些 token 表示模型内部当前激活中可 verbalize 的概念。

---

### Step 5：求 J-space component

用 J-lens vectors 作为 dictionary，对 activation 做稀疏非负分解：

```text
h_J, h_nonJ, active_tokens, coeffs = sparse_decompose(h, D_ℓ, k)
```

其中：

```text
h_J = Σ_i a_i v_i
h_nonJ = h - h_J
```

---

## 10. 干预算法

J-lens 不只是读取工具，还可以用于 causal intervention。

### 10.1 Steering：注入概念

给定 token `x` 的 J-lens vector `v_x`：

```text
h ← h + α v_x
```

解释：

- `α > 0`：增强该概念。
- `α < 0`：抑制该概念。

例如，把 `spider` 的方向注入中间层，观察模型后续回答是否更倾向于蜘蛛相关内容。

---

### 10.2 Ablation：移除概念

移除某个概念方向：

```text
h ← h - proj_{v_x}(h)
```

其中：

```text
proj_{v_x}(h) = ((h · v_x) / (v_x · v_x)) v_x
```

也可以对多个活跃 J-space vectors 批量 ablate：

```text
for v in active_j_space_vectors:
    h ← h - proj_v(h)
```

用途：测试某些 verbalizable concepts 是否对最终输出有因果作用。

---

### 10.3 Coordinate Patching：替换概念

如果想把 source concept `s` 替换为 target concept `t`：

```text
v_s = J-lens vector for source token
v_t = J-lens vector for target token
V = [v_s, v_t]
```

先求局部坐标：

```text
c = V† h
```

其中 `V†` 是 pseudoinverse。

交换坐标：

```text
c' = swap(c)
```

写回 activation：

```text
h_patched = h + V(c' - c)
```

这会尽量只修改 `span{v_s, v_t}` 内的成分，保留其他方向不变。

---

## 11. 简化伪代码

```python
# Pseudo-code only


def estimate_jacobian_lens(model, prompts, layer_idx):
    jacobians = []

    for prompt in prompts:
        cache = model.forward_with_cache(prompt)

        for t in source_positions(prompt):
            h_source = cache.residual[layer_idx][t]

            for t_future in future_positions(t, prompt):
                h_final = cache.residual[-1][t_future]

                # Compute local Jacobian:
                # ∂h_final / ∂h_source
                J = autograd_jacobian(h_final, h_source)
                jacobians.append(J)

    J_layer = mean(jacobians)
    return J_layer



def j_lens_readout(model, J_layer, h_layer):
    h_final_approx = J_layer @ h_layer
    h_final_approx = model.final_norm(h_final_approx)
    logits = model.W_U @ h_final_approx
    return top_tokens(logits)



def j_space_decompose(h, j_lens_vectors, k=25):
    """
    Approximately solve:
        min ||h - Σ a_i v_i||²
        subject to a_i >= 0 and at most k active vectors
    """
    active_vectors, coeffs = gradient_pursuit(
        target=h,
        dictionary=j_lens_vectors,
        sparsity=k,
        nonnegative=True,
    )

    h_j = sum(a * v for a, v in zip(coeffs, active_vectors))
    h_nonj = h - h_j

    return h_j, h_nonj, active_vectors, coeffs



def projection(h, v):
    return ((h @ v) / (v @ v)) * v



def steer_concept(h, v_token, alpha):
    return h + alpha * v_token



def ablate_concept(h, v_token):
    return h - projection(h, v_token)



def swap_concepts(h, v_source, v_target):
    V = stack_columns([v_source, v_target])
    c = pseudoinverse(V) @ h
    c_swapped = swap_two_coordinates(c)
    return h + V @ (c_swapped - c)
```

---

## 12. 实现 checklist

如果要在 open-weight decoder-only Transformer 上实现简化版：

- [ ] 选择模型，例如 Llama / Qwen / GPT-style decoder-only model。
- [ ] hook 每层 residual stream `h_{ℓ,t}`。
- [ ] 收集 calibration prompts。
- [ ] 对每层估计平均 Jacobian `J_ℓ`。
- [ ] 用 `J_ℓ` 构造 J-lens readout。
- [ ] 对任意 prompt 的中间层 activation 输出 top tokens。
- [ ] 构造 J-lens vectors 作为 concept dictionary。
- [ ] 实现 sparse nonnegative decomposition 得到 J-space component。
- [ ] 实现 steering / ablation / coordinate patching 做因果验证。

---

## 13. 算法直觉

可以把 decoder-only Transformer 的 residual stream 看作一块逐层更新的共享黑板：

```text
h_0 → h_1 → h_2 → ... → h_L
```

最终层 `h_L` 直接决定模型会说什么。

Jacobian Lens 问的是：

```text
如果我在中间层 h_ℓ 上沿某个方向轻轻推一下，
最终模型更可能说出哪些 token？
```

对大量上下文平均后，这些方向就形成较稳定的“可语言化概念方向”。

因此：

- **J-lens**：读取中间层 activation 中 verbalizable concepts 的 lens。
- **J-space**：由这些 verbalizable concept directions 组成的稀疏工作区。
- **Steering / ablation / patching**：对这个工作区做因果干预的方法。

---

## 14. 核心贡献总结

1. **提出 Jacobian Lens**  
   用平均 Jacobian 把中间层 residual stream 映射到最终层 verbalizable token space。

2. **定义 J-space**  
   把 J-lens token vectors 的稀疏非负组合定义为模型中的 verbalizable representation space。

3. **支持因果干预**  
   通过 steering、ablation、coordinate patching，可以测试某些内部概念是否影响模型最终输出。

4. **连接 global workspace 直觉**  
   J-space 具有容量有限、可读取、可调制、可广播、可参与推理等特征，类似一种模型内部的“全局工作空间”。

---

## 15. 最短总结

```text
Jacobian Lens = average Jacobian + unembedding readout
J-space = sparse nonnegative combination of J-lens token vectors
核心实验 = 读取、注入、删除、替换 J-space 中的概念，观察模型行为变化
```
