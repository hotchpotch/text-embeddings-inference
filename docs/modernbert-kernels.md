# ModernBERT FP16 reranker kernels

This branch starts from Hugging Face TEI main at `069ceab366d7cab14cfd40dbcb0d58c93f892583` and packages the A5 CUDA implementation. Model architecture, weights, layers, tokenization, candidates, and attention visibility are preserved. No quantization, CUDA Graph cache, or approximate attention is added.

## Computation

- Fused residual addition and LayerNorm carry the residual separately between FlashModernBERT layers.
- CLS pooling in a final global-attention layer computes only the required CLS queries and final-layer outputs. Keys/values still cover every input token. Raw-token, mean-pooling, and non-global final-layer requests retain the full-query path.
- Dual GeGLU uses FP16 inputs/weights and FP32 accumulation; it rounds the two projection outputs and GELU output back to FP16 before their product. It uses the original tanh GELU expression rather than the experimental hardware-tanh approximation.
- The original PR #897 split-projection GELU path remains for eligible CUDA paths that do not use the fused FP16 implementation. CPU and non-GELU models retain their original projection behavior.

## Embedded kernel dispatch

Bundles under `backends/candle/kernels/modernbert/` support original hidden/intermediate dimensions **256/1024** and **512/2048**. Selection uses dimensions, not a model-name match. It requires CUDA FP16, GELU, bias-free Wi, and a runtime GPU capability of SM89 or newer. Other shapes/devices use the generic MLP path.

| Runtime activation rows | Dual GeGLU kernel |
| --- | --- |
| 1–32 | small |
| 33–128 | medium |
| 129–2047 | generic projection plus fused GeGLU |
| 2048 and above | large |

These names describe CUDA tile choices, not model sizes. Input tensor contents and row counts are passed on every invocation; only executable kernels are reused. Each invocation allocates its output, and a device-module lookup plus a synchronized double check protects first loading. Changing queries/documents does not reuse earlier scores or require recompilation per text.

The bundles retain the SM89 PTX instructions used for the reference measurements; debug locations and DWARF sections were stripped for distribution. A CUDA driver loads/JITs a variant when first used. Native TEI warmup does not necessarily exercise every variant: with a 36,864-token budget and 8,192-token maximum input it builds five sequences, so the final CLS MLP uses small; a later 120-candidate batch may first exercise medium. This is a plausible explanation for the observed first-request overhead, not a measured attribution of individual JIT intervals.

The embedded PTX declares ISA version 8.7; the NVIDIA driver must support loading that PTX version. Validation used driver 595.99.02 with CUDA Toolkit 12.9.

SM89 includes NVIDIA L4, but L4 throughput and startup have not been measured. The reference GPU was SM120; compile the rest of TEI for the actual serving GPU as required by upstream. Older architectures do not load the embedded SM89 bundle.

## Request grouping

`TEI_RERANK_REQUEST_BATCH=1` finishes tokenization of one HTTP rerank request before appending its candidates together to the existing queue. It does not skip tokens/documents, bypass batch token/request limits, or add a batching delay. This feature is opt-in and was enabled for baseline, PR, and A5 in the reference measurements. Standard routing remains available by leaving the variable unset. Queue scheduling can change batch boundaries and small floating-point/ranking differences.

## Rebuild the PTX

Inference does not require Triton. Regeneration uses Python with `triton==3.5.1`, and the checked-in generator:

```sh
python -m pip install triton==3.5.1
python backends/candle/kernels/modernbert_geglu.py --arch 89 --model-size xsmall \
  --output-dir backends/candle/kernels/modernbert/xsmall
python backends/candle/kernels/modernbert_geglu.py --arch 89 --model-size base \
  --output-dir backends/candle/kernels/modernbert/base
```

The metadata describes the ABI, shared memory, model dimensions, and tile configuration. `manifest.json` records SHA256 hashes of the distributed bundles. Test any regenerated kernels numerically before replacing them; compiler/toolchain changes can change generated PTX.

## Validation

CUDA tests require an SM89-or-newer GPU and a matching CUDA build:

```sh
cargo test --release --locked -p text-embeddings-backend-candle -p text-embeddings-router --lib \
  --features text-embeddings-router/candle-cuda modernbert -- --test-threads=1
cargo test --release --locked -p text-embeddings-backend-candle -p text-embeddings-router --lib \
  --features text-embeddings-router/candle-cuda geglu -- --test-threads=1
cargo test --release --locked -p text-embeddings-backend-candle -p text-embeddings-router --lib \
  --features text-embeddings-router/candle-cuda fused_layer_norm -- --test-threads=1
cargo test --release --locked -p text-embeddings-core --lib queue:: -- --test-threads=1
```

The dual-kernel test exercises both model dimensions, sliced tensors, dispatch boundaries, fallback shapes, and numerical agreement with the FP16 reference. The final-CLS test compares unequal-length sequences against full attention. Queue tests cover token/request limits and preparation behavior. Reference throughput and its conditions are summarized at the top of the root README; they are not a guarantee for every model or query distribution.

## Standalone branch validation (November 2, 2026)

The packaged CUDA 12.9 release binary passed the CUDA/CPU MLP, unequal-length final-CLS, dual-GeGLU, residual-LayerNorm, and queue tests (11 tests total), plus a CPU-only backend build check. Both model dimensions were exercised at kernel dispatch boundaries. All six distributed PTX kernels also assembled for SM89; executable PTX matched regeneration with Triton 3.5.1 after excluding debug metadata.

A separate packaging check replayed every original request against the saved A5 implementation and this standalone branch: two models, two implementations, two inputs, three full passes, **24 passes / 49,416 request appearances / 2,638,680 pair appearances**. This check used 30 seconds of preheat, concurrency 8, workers 8, and one execution order; it is not the forward/reverse baseline/PR benchmark shown in the root README.

| Model | Replay | Saved A5 (pairs/s) | Standalone branch (pairs/s) | Top-1 changes vs saved A5 |
| --- | --- | ---: | ---: | --- |
| xsmall | limit20 | 14,261.87 | 14,179.28 | 0/2,061 in each of 3 passes |
| xsmall | limit100 | 14,326.83 | 14,182.34 | 0/2,057 in each of 3 passes |
| base | limit20 | 3,972.51 | 3,963.48 | 0/2,061 in each of 3 passes |
| base | limit100 | 4,068.45 | 4,066.79 | 0/2,057 in each of 3 passes |

Every response preserved candidate and token counts and had finite scores. Scores were not bit-identical; this is a rank-preservation check on a private production replay, not a new relevance evaluation. These measurements used the same Blackwell GPU as the reference runs, not NVIDIA L4.
