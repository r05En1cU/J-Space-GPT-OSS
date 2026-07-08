# Validation Notes

## Static validation

The implementation is syntactically self-contained and exposes these CLIs:

```bash
python jspace_gpt_oss.py inspect-model
python jspace_gpt_oss.py build-dictionary
python jspace_gpt_oss.py readout
python jspace_gpt_oss.py decompose
python jspace_gpt_oss.py intervene
```

## Local dependency status observed during implementation

The default/base Python in this workspace does not currently import `torch`, so the math smoke test and CLI import test cannot run from base Python. Use the project GPU environment that has PyTorch installed, for example:

```bash
conda activate StarPhoton
cd /ai/mount/stlsy/workspace/J-Space-GPT-OSS
python smoke_math_test.py
python -m py_compile jspace_gpt_oss.py
```

Then run the model-level reproduction:

```bash
bash run_minimal_repro.sh
```

## Expected first-pass artifacts

- `gpt_oss_20b_jspace_dictionary.pt`
- `readout_spider_layer12.json`
- `decompose_spider_layer12.json`
- `steer_spider_layer12.json`

## Success checks

- `smoke_math_test.py` prints `ok`.
- `inspect-model` reports a valid decoder block path and layer count.
- `build-dictionary` logs each sampled layer and writes a `.pt` dictionary.
- `readout` returns semantically related top concepts for prompts such as `A spider builds a`.
- `decompose` returns a sparse nonnegative active-token list and a nonzero explained fraction.
- `intervene --mode steer --token " spider"` changes generation toward the target concept relative to an unsteered baseline.
