# RQ1 Phase 1 执行规格(预注册级)—— (b′) go/no-go 门

> 上位文档 `PLAN_RQ1_moe_routing.md`。本文件把 Phase 1 钉到"跑之前每个指标/基线/阈值都定死"。
> 所有阈值、样本量、判据在**看数据前**写定;跑完只填数,不改判据(改判据=HARKing,须显式标注为探索性)。

## 0. 记号与已知(据 Phase 0)
- M_ℓ = W_U·diag(g)·J_ℓ(readout 矩阵)。Phase 0 实测:M_ℓ stable-rank≈1.5–2,J_ℓ 近满秩含恒等通路。
- router 判别方向 δ_ij = normalize(γ⊙(w_i−w_j)),γ=post_attention_layernorm.weight,w=router.weight 行。
- 两 gate 口径:**G-diff**(gate 可导,现有 J)与 **G-det**(JVP 路径对 router_scores 做 detach 的 J)。

## 1. 层选择
- **主结果层**:8,12,16,20(中后层,routing 决策实质、J≠I)。
- **控制层**:0,4(浅层 J≈I,预期"都在工作区"是恒等伪影,仅作对照,不进主判据)。
- 结论要求在主结果层 **跨层稳健**(4 层里 ≥3 层同向),单层结果不算。

## 2. 语料与估计/评估集切分(防软循环)
- 语料:paper_probes.txt 的 flexible-generalization + probe-swap 子集(已验真),取 ≥40 条 prompt。
- **切开**:随机分 A/B 两半(种子固定)。**平均 J_ℓ 只在 A 上估计**;所有 δ_ij 选取、百分位、投影评估**只在 B 上做**。A/B 不重叠。
- 竞争 expert 对定义:B 上每个 token 位置,router top-2 gate 差 <0.1(近平票)的对,取 δ_ij。目标 ≥300 个 (pos, i, j) 竞争对/层。

## 3. 主判据 A —— δ_ij 的 readout 能量百分位(能量腿,必要非充分)
- 量:E(δ_ij) = ‖M_ℓ δ_ij‖。基线:R 个匹配范数随机单位方向 u 的 ‖M_ℓ u‖ 分布(R=1000,种子固定)。
- 报告:每个 δ_ij 的百分位 = P(‖M u‖ < ‖M δ_ij‖);对全部竞争对取**中位百分位**,按层。
- **两口径都算(G-diff / G-det),必须同向。** 分开报,不合并。
- 判据数值:中位百分位 **>90**(能量在)/**<10**(能量不在)为极端;40–70 为中间地带。

## 4. 主判据 B —— 路由决策方差谱分解(结构腿)
- 量:B 上各 token 的 router logit gap 向量(或 top-1 vs top-2 logit 差的梯度方向),投到 M_ℓ 的 top-r 右奇异子空间,算被解释方差占比 f(r)。
- 基线(必配,防维度伪影):**秩匹配随机 r 维子空间** + **激活 PCA top-r 子空间**,各自的 f_rand(r)、f_pca(r)。
- 报告:f(r)−f_rand(r) 的**曲线**(r 从 1 扫到 ~50,据 Phase 0 M 有效维小),**禁止单点 r 二分**。
- 判据:存在 r≪d(如 ≤20)使 f(r)>0.7 且 f(r)−f_rand(r) 显著(曲线明显高于两基线),为"结构在";曲线贴基线为"结构不在"。

## 5. 主判据 C —— 因果投影(因果腿)
- 操作:把 B 的 h_ℓ 投到 M_ℓ top-r 子空间(P_r h)喂 router,测 top-k 路由保留率(与原路由 top-k 集合 IoU);对照投到补空间 (I−P_r)h,及**秩匹配随机 r 维投影**。
- 判据:top-r 投影保留率显著 > 随机投影(路由在 J-space);补空间投影保留而 top-r 摧毁(路由不在)。report 随 r 曲线。
- 注意 margin:top-k 靠 margin,近平票路由易被小扰动翻,保留率要按 gate_gap 分档报,别混。

## 6. 主判据 D —— 可语言化直测 (f)(语义腿,能量在也要过这关)
- 操作:δ_ij 过 lens = M_ℓ δ_ij ∈ R^vocab,取 top-20 token。判连贯性:LLM-judge 打分(0-2:是否构成可解释的语义/概念簇)。基线:R'=100 个随机单位方向的 top-20 token 连贯性分布。
- 判据:δ_ij 的连贯性显著 > 随机方向基线(可语言化)/ 与随机无异(仅几何非语义)。
- **缺 D 则整条 RQ 只是"readout 几何"非"全局工作区"** —— A/B/C 过、D 不过 = 负结果(readout 可见但不可语言化,反例)。

## 7. within-token 控制(防 token-ID 混淆,fable 标记最可能烂尾)
- 选在 B 中跨 ≥5 个不同上下文出现的 token 类型(如 " the"" of"),目标 ≥15 个类型。
- 量:同一 token 类型跨上下文的**路由变化方向** Δroute(context 间 top-k / gate 变化对应的 h 方向差),测其 readout 能量百分位(同判据 A)+ 可语言化(同判据 D)。
- **退出条件(预注册)**:若 within-token 路由方差被噪声主导(跨上下文路由几乎不变,或 Δroute 百分位与随机无异)→ **within-token 信号不足,RQ1 正结果降级为"仅 token-ID 层面",转 RQ-CoT**。这条不满足,即使 A–D 全过也不能声称"路由在工作区"是新事实。

## 8. Go / No-Go 决策规则(看数据前定死)
**GO(进 Phase 2)** 需同时:
1. 主判据 A 中位百分位极端(>90 或 <10),主结果层 ≥3/4 同向,**两 gate 口径一致**;
2. 主判据 B 曲线显著离两基线(结构在或明确不在);
3. 主判据 C 因果与 A/B 同向;
4. **within-token(§7)有信号**(Δroute 百分位极端,非噪声);
5. D 明确(可语言化 or 明确不可语言化——后者是有价值的负结果)。

**NO-GO / 降级转 RQ-CoT** 若:A 中位百分位落 40–70 中间地带,**或** 两 gate 口径矛盾,**或** within-token 被噪声吃(§7 退出),**或** B/C 曲线贴基线无结构。

**方向无关**:正结果(在)和干净负结果(不在/不可语言化)都算 GO——负结果是 headline。只有"中间地带 + within-token 噪声主导"才降级。

## 9. 交付物
- `rq1_phase1_bprime.json`:A/B/C/D + within-token,分层、分 gate 口径、含所有基线分布。
- 决策留档:对照 §8 逐条 GO/NO-GO,哪条卡住如实写。
- 图:δ_ij 能量百分位分布(vs 随机)、方差分解 f(r) 三线对比、因果保留率 vs r。

## 10. 工程注意
- G-det 口径要新写:JVP 路径里对 router_scores/routing_weights 做 detach 的模型变体(hook 或 patch),别动 G-diff 主路径。
- 实体化 M_ℓ top-r 子空间:随机化 SVD(Phase 0 脚本已有能力),r 取到 ~50 够(M stable-rank~2)。
- 两口径 × 4 主层 × A/B/C/D 计算量不小,先在 1 层(16)把 A–D + within-token 全跑通做管线冒烟,再铺全层。
