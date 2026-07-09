# RQ1 执行计划(草案)——MoE 路由 × J-Space

## 命题
gpt-oss-20b 的 MoE 路由信息是否**显式存在于 J-Space**(论文意义上"可语言化的全局工作区")里,
还是活在 J-space 的补空间/不可语言化处。这是 dense 模型复现没有的可测对象:top-k 路由离散
不可导,我们的 J_ℓ 是"固定当前路由"的局部雅可比,**路由方差本身可测**。

## 现有资产(readout-full 线闭环后)
- `readout-full`:全 vocab J-lens 分数,JVP(J·v),经 mutation-test 验证的回归门,全层过门(0-23)。
- 路由 dump 能力:已能对任意 prompt/token/层 dump top-k expert 索引 + gate(见 layer16 诊断脚本)。
- 严格口径:J_ℓ 到 pre-norm 残差,g 折入,固定路由语义(与论文 VJP 一致)。
- MoE patch(可 scoped 还原)+ 权重 lift 到 fp32/fp64 的能力(判别数值 vs 语义差异)。

## 三个候选设计(判别力/成本/陷阱)
### (a) J-sample 聚类:按激活 expert 集合分组,比组内/组间 Jacobian 方差
- 测:路由模式是否在 J 里留下可分结构。
- 判别力:中(描述性,"有结构"不直接等于"路由在工作区")。
- 陷阱:expert 集合是高维离散标签,聚类相似度度量的选择会主导结论;组间方差大也可能只是
  输入分布差异,非路由本身。

### (b) h_J vs h_nonJ 分别 probe 下一层 router(主命题候选)
- 把 residual h_ℓ 分解为 J-space 内投影 h_J 与补空间 h_nonJ,分别喂下一层 router,看谁驱动路由。
- 若路由主要由 h_nonJ 驱动 → **"路由不在全局工作区"** 强命题(negative 也强)。是 Recall-Space pilot。
- 判别力:最强。成本:高。
- **核心陷阱(待 fable 裁)**:h_J/h_nonJ 的"J-space"如何定义才不循环?J-space 是 W_U·diag(g)·J_ℓ
  的行张成的子空间(readout 可见方向),还是别的?投影算子定义错 → 整个结论无意义。

### (c) steer 一阶预测 vs 实际,按是否翻转路由分组(de-risk 候选)
- 用 J-lens 向量 steer,记录一阶预测 Δlogit,对比实际 Δlogit;按该 steer 是否翻转了下游路由分组。
- 测:固定路由 J_ℓ 的适用边界(路由不翻时一阶应准,翻转时应系统性偏离)。
- 判别力:中强。成本:最低(接现有 steer + 路由 dump)。
- **核心陷阱(待 fable 裁)**:我们的 MoE patch 是"路由固定"的,一阶预测本就不含 expert 翻转效应,
  所以"翻转组一阶偏大"是否**同义反复**(by construction 必然),而非可证伪发现?若是,(c) 的判别力虚高。

## fable-advisor [max] 裁决(2026-07-09 定,推翻我原倾向)

**总裁决:c→b 顺序基于错误成本模型。gpt-oss router 是线性层(h→RMSNorm→W_r→top-k softmax),
RMSNorm 正标量缩放不改 router logit 排序 ⇒ top-k 选择是 h 的精确分片线性函数。所以 (b) 大部分
坍缩成闭式线性代数(b′),成本反低于 (c),de-risk 理由消失。(c) 原版 headline 近不可证伪,降为
Phase 2 效度研究。(a) 降为附图。**

### Phase 0 —— 写任何代码前必须钉死的两个实现事实(预注册)
- **(0-1) MoE patch 里 gate 权重梯度是否 detach? —— 已确认(2026-07-09,读 transformers 源码):gate 可导,top-k 集合冻结。**
  证据(`.../transformers/models/gpt_oss/modeling_gpt_oss.py`):`GptOssTopKRouter.forward`(L132-135)`router_scores=softmax(topk(F.linear(h)))`,
  gate 无 detach、对 h 可导;`GptOssExperts.forward` 的 `with torch.no_grad()`(L97) 只裹 one-hot `expert_mask`
  构造,`routing_weights` 乘到输出(L116)在 no_grad 外、梯度保留。即 fable 说的"标准写法:冻结 top-k 集合但 gate 值可导"。
  **后果坐实:J_ℓ 已通过 gate 路径(∂output/∂routing_weights·∂routing_weights/∂h)把 router 输入方向部分收进 J-space
  ⇒ "路由在工作区" by construction 部分偏向"在"。对策(现为必做非可选):Phase 1 gate-detach 与 gate-differentiable
  两口径各算一遍 J_ℓ,结论两口径一致才算稳。** gate-detach 口径需在 readout/JVP 路径对 router_scores 做 detach 变体。
- **(0-2) 平均 J vs 逐 token J 错位。** 路由是逐 token 决策,平均 J 抵消跨 token 符号翻转分量——
  router 方向可能在平均 J 低奇异值区却在逐 token J 响亮(反例:某方向对名词+/动词−,平均近零但正是
  分专家的方向)。**对策:主结果用平均 J(合论文口径),预注册逐 token 子样本核对;矛盾即发现。**
- **J 估计集 / 评估集必须切开**:同批 prompt 既估平均 J 又评路由 = 软循环过拟合。

### (b) 精确子空间定义是致命缺陷(必须在写代码前改掉)
M_ℓ = W_U·diag(g)·J_ℓ:W_U 列满秩 + J_ℓ 含恒等通路近满秩 ⇒ row(M_ℓ)=整个 R^d ⇒ **精确补空间=零空间,
h_nonJ≡0,(b) 原版坍成"一切都在工作区"空话**(不是我担心的补空间平凡主导,是补空间平凡为空)。
**正确定义必须谱性/软性**:M_ℓ 右奇异向量按奇异值排 top-r;但 r 是自由参数,**必须报 r 扫描曲线 +
秩匹配基线(随机 r 维子空间 / 激活 PCA top-r),不做单点二分**。无参版:‖M_ℓ δ‖ 对匹配范数随机方向
分布的百分位。语义缺口:M 下能量大 ≠ 可语言化,工作区隶属 = 能量 × 读出连贯性的合取(见设计 f)。

### Phase 0 实测结果(2026-07-09/10,手动上机,layer 4+16,prompt="The legal contract states that the buyer")
- **核查1 gate**:已确认可导(见上)。
- **核查3 J_ℓ 谱**:J_ℓ **近满秩 + 恒等通路实锤**——随机方向 σ 中位数 L4=2.19/L16=1.01、最小 0.62–0.86 无近零,400 探测方向全在阈值上;stable_rank≈5.4/5.7(仅计放大方向,主体≈I)。**⇒ 精确补空间≈零,(b) 原版坍塌确认,必须谱性定义。** M_ℓ stable_rank≈1.5–2(σ 4340→陡降),**readout 可见子空间有效维极小,(b′) 的 r 扫描窄(~几十),子空间定义干净**。
- **核查2 平均vs逐token**:无抵消——‖平均‖/mean‖逐token‖ 随机 0.64–0.66、**router 方向 0.66–0.70(不降反升)**,sign_fraction=1.0。**⇒ 平均 J 公允,逐 token 核对维持子样本不升主结果。** 附:router 判别方向 J 响应(12–20)是随机(~2.3)的 5–8 倍,坐在 J 放大子空间——对"路由在工作区"初步利好,但测的是 ‖J·δ‖ 非 ‖M·δ‖、n=1 冒烟,非结论。
- **待补**:核查2 只 1 个 prompt/层,Phase 1 需几个 prompt 复核;正向信号需换成 ‖M·δ‖ 的百分位判别才算数。

### 采纳的分阶段结构(替换原 c→b)
- **Phase 0(几天)**:钉死 0-1/0-2 + 实体化平均 J_ℓ(d=2880,每层 2880 次 JVP 累加,A100 天级)+ 看谱。
- **Phase 1(≈1 周)= go/no-go 门**:(b′) 静态对齐 + 路由决策方差谱分解 + 精确投影因果干预 + (f) 可语言化
  评分。判据:能量百分位极端(>90 或 <20)且 within-token 控制有信号 → 进 Phase 2;中间地带(40–70、
  无鲜明结构)且 within-token 被噪声吃掉 → 降级止损,算力转 RQ-CoT。
- **Phase 2**:(e) 双向因果 steer 矩阵(工作区→路由 / 路由方向→工作区 的 2×2 耦合),吸收剂量-响应版 (c)。

### 新增设计(fable 提出,判别力≥原 b)
- **(b′)** 闭式对齐 + 方差分解 + 精确投影干预 = (b) 的正确形态,首发。
- **(e)** 双向 steer 矩阵:沿 top-J 方向 steer 测路由翻转率 vs 随机;反向沿 δ_ij steer 测 lens/logit 响应。
  "路由方向 steer 大幅改 logits 但 lens 读不出" = 对论文强命题的直接反例,判别力最高单一结果。
- **(f)** 可语言化直测:δ_ij 过 lens 取 top tokens,LLM-judge + 随机方向基线评连贯性。**最忠实论文
  "verbalizable" 语义,三原设计都缺它,几乎免费。缺 (f) 整条 RQ 只是"readout 几何"非"全局工作区"。**
- **(c) 改造版(剂量-响应)**:扫 steer 幅度 α 找路由翻转阈值 α*,看线性化误差在 α* 拐点——同 token 内
  跨 α* 前后对比,隔离"翻转本身"vs"近边界 token 普遍高曲率"混淆。测的是 expert 边界功能多样性(Δslope),
  是 lens 效度研究,**不是 RQ1 证据**,给 (b′) 结论垫底。

### 最小可证伪单元
- **正结果(路由在工作区)三条同时成立、跨层(0-23)稳健**:①竞争 expert 对判别方向 δ_ij 的 lens 能量
  在匹配基线 >90 百分位;②路由决策方差 >70% 落在 r≪d 的 top-J 方向(对照秩匹配基线,曲线形状不依赖单点 r);
  ③因果:top-r J-space 投影保留路由率显著 > 随机投影,**且** δ_ij lens 读出 token 连贯性显著 > 随机基线。
- **负结果**:补空间投影保留路由、J-space 投影摧毁路由,r/层扫描稳定,gate 两口径一致。

### 似是而非、什么都没证明的结果清单(预注册,禁止事后当发现)
1. 单点 r 二分比较(维度伪影);2. capacity 不匹配 probe 差异;3. (c) 原版组间比较无剂量-响应控制;
4. **token-identity 混淆(最危险)**:MoE 路由≈token ID 是已知结果,token ID 平凡可语言化;"路由在工作区"
   若只是重发现"路由≈token ID、token ID 可读出"则增量为零。**必须 within-token-type 跨语境路由方差控制**——
   这块可能噪声主导,是最可能烂尾处,预注册"within-token 信号不足则降级"退出条件;5. 任何单层结果(浅层
   J≈I 时"都在工作区"是恒等映射伪影,浅层结果打折)。

### 增量论证(现在立起来,防"dense 换 MoE 重跑"指控)
防御点:dense 复现没有任何**带 ground truth 的离散内部决策变量**;路由是第一个能精确定位"模型内部控制
决策相对工作区在哪"的对象——新问题类型非新数据集。够格增量:①**负结果即 headline**(路由对 logits 有
实质因果效应但 lens 不可见 = 对"可语言化表示构成全局工作区"强命题的具体反例,比正结果值钱);②方法增量
(线性化 lens 在离散不连续点的效度边界 + 带匹配基线的谱性工作区隶属度,可推广到 early-exit/检索/任何 MoE);
③正结果仅当 within-token 语境路由可语言化才算新事实。中间地带 + within-token 被噪声吃 → 降附录转 RQ-CoT。
