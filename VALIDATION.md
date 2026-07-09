# 验证说明

## 静态验证

本实现语法自洽，暴露以下 CLI 子命令：

```bash
python jspace_gpt_oss.py inspect-model
python jspace_gpt_oss.py build-dictionary
python jspace_gpt_oss.py readout
python jspace_gpt_oss.py decompose
python jspace_gpt_oss.py intervene
```

## 本地依赖状态

本工作区的 base Python 不能 import `torch`，因此数学 smoke test 与 CLI import 测试无法在 base 环境运行。请使用装有 PyTorch 的项目 GPU 环境，例如：

```bash
conda activate StarPhoton
cd /ai/mount/stlsy/workspace/J-Space-GPT-OSS
python smoke_math_test.py
python -m py_compile jspace_gpt_oss.py
```

随后运行模型级复现：

```bash
bash run_minimal_repro.sh
```

## 首次运行预期产物

- `gpt_oss_20b_jspace_dictionary.pt`
- `readout_spider_layer12.json`
- `decompose_spider_layer12.json`
- `steer_spider_layer12.json`

## 成功判据

- `smoke_math_test.py` 打印 `ok`。
- `inspect-model` 报告有效的 decoder block 路径与层数。
- `build-dictionary` 逐层打印日志，并写出 `.pt` 字典；token 向量严格取为 `W_U J_ℓ` 的行（对 pre-norm 的 `h_final` 做 VJP）。
- `readout` 对 `A spider builds a` 等 prompt 返回语义相关的 top concepts；默认用原始点积 `⟨v_t, h_ℓ⟩` 打分（`--cosine` 可切换为余弦相似度）。
- `decompose` 返回稀疏非负的活跃 token 列表与非零的 explained fraction。
- `intervene --mode steer --token " spider"` 相对未干预基线，使生成更倾向目标概念。
