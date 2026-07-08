# J-Space / Jacobian Lens Reproduction on GPT-OSS-20B

This directory implements a practical reproduction path for the `ALGORITHM.md` description of **Jacobian Lens** and **J-space** on a HuggingFace decoder-only model such as GPT-OSS-20B.

## What is implemented

The paper algorithm defines an average Jacobian from an intermediate residual stream to the final residual/logit space. For a 20B model, this implementation uses the equivalent and tractable token-vector form:

```text
v_{layer, token} = E[ d logit_token(target_position) / d h_layer(source_position) ]
```

That is the J-lens token direction needed for:

- **J-lens readout**: score a layer activation against verbalizable token directions.
- **J-space decomposition**: sparse nonnegative pursuit over token vectors.
- **Steering**: add a concept vector to a residual stream position.
- **Ablation**: remove the projection on a concept vector.
- **Coordinate patching**: swap two concept coordinates in the local J-space span.

The implementation intentionally avoids materializing full `d_model x d_model` Jacobians for every layer. It computes vector-Jacobian products directly through a layer-output hook, cutting the graph at the selected layer so early blocks do not need to keep gradients.

## Files

- `ALGORITHM.md` — conceptual algorithm notes.
- `jspace_gpt_oss.py` — runnable implementation and CLI.
- `calibration_prompts.txt` — default calibration prompts.
- `candidate_concepts.txt` — default verbalizable concept candidates.
- `requirements.txt` — Python dependencies.
- `smoke_math_test.py` — lightweight math-only checks for pursuit/projection/patching.

## Installation

Use a GPU environment suitable for the target model. For GPT-OSS-20B, prefer BF16 on a large GPU or multi-GPU device map. 4-bit loading is exposed for constrained runs, but exact gradient behavior depends on the local `bitsandbytes` / `transformers` stack.

```bash
pip install -r requirements.txt
```

If the model is already cached locally, add `--local-files-only` to commands.

## Inspect the model

```bash
python jspace_gpt_oss.py inspect-model \
  --model-id openai/gpt-oss-20b \
  --torch-dtype bfloat16 \
  --device-map auto
```

This confirms the decoder block path and number of layers.

## Build a J-lens dictionary

Start small, then scale. A first useful run samples a few layers, one source/target pair per prompt, and the candidate concept list:

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

For fuller reproduction, increase:

- `--layers all`
- prompt count
- candidate concept count
- `--max-pairs` with `--position-mode causal-window` or `all-same`

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

The output is JSON with ranked token concepts and scores.

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

The result reports active token IDs/texts, nonnegative coefficients, residual norm, and explained fraction.

## Causal interventions

Steer toward a concept:

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

Ablate a concept:

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

Patch two concept coordinates:

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

## Reproduction protocol

Recommended experimental table:

| Stage | Goal | Minimal setting | Scaled setting |
|---|---|---:|---:|
| Dictionary | Estimate J-lens token vectors | 6 layers × 8 prompts × 40 concepts | all layers × 100+ prompts × 1k+ concepts |
| Readout | Verify verbalizable concepts | top-20 cosine scores | compare vs logit lens baseline |
| Decomposition | Estimate J-space sparsity | `k=25` | sweep `k ∈ {5, 10, 25, 50}` |
| Intervention | Causal effect | steer/ablate selected concepts | paired prompts + effect-size table |

Core metrics to log:

- readout top-k overlap with human-expected concepts;
- decomposition explained fraction vs `k`;
- steering delta in target concept logits;
- ablation drop in target concept logits;
- patching success rate on paired source/target prompts.

## Notes and caveats

- This implementation estimates **token-vector J-space** by VJP, not a stored full average Jacobian matrix.
- Candidate vocabulary size controls runtime linearly. Use a focused concept list first, then expand.
- Multi-token candidate strings are represented by their final token ID, which matches the token-level definition but should be documented in analysis.
- Greedy intervention generation disables KV cache for correctness and simplicity; it is intentionally slow.
- If `openai/gpt-oss-20b` is not the local model ID in your environment, pass the correct HuggingFace path with `--model-id`.
