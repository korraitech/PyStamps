# Step 2: Handwired on the shared-baseline trial search

**Result:** Handwired can execute this PyStamps calculation on the hardware GPU. Fusing correlation and winning-trial selection reduced the measured 300,000-pixel search from **10.676 seconds to 79.7 ms**. However, **batched NumPy at the original precision took 58.8 ms**, selected every original winner in this dataset, and avoids a JavaScript/GPU integration. Handwired's FP32 computation changed one near-tied winner and its subsequent refinement. Recommend batched FP64 for the next production validation, keeping Handwired experimental.

Production source and requirements are unchanged. This is a feasibility benchmark, not a deployed optimization or full-scene validation.

## Scope and representation

Step 2 calls `ps_topofit` 300,000 times to calibrate random-phase coherence, sharing the same baseline vector (`src/step/step_2_ps_estmgm.py:112-118`). Each call rebuilds the same trial matrix. This experiment covers the trial correlation and winning-trial selection in `src/step/ps_topofit.py:10-22`. Phase refinement and least squares remain the original calculations.

For complex inputs `x` and fixed trial phasors `W`, correlations are `x @ W`. Their real representation is `[Re(x), Im(x)] @ [[Re(W), Im(W)], [-Im(W), Re(W)]]`. The 60-interferogram, 33-trial fixture becomes a 120-input, 66-output fixed-weight affine network. There is no training. Trial weights are computed once in NumPy and supplied to the actual upstream Handwired `Circuit` builder.

The network uses partial sums of at most 24 terms to limit generated expression sizes. It contains 332 live operations and two layers. The optional fused selector adds squared magnitudes and an exact comparison tree, returning the first winning index and score in a single GPU dispatch: 1,334 live operations, 115 circuit layers. These layers are statements inside one shader, not 115 GPU dispatches. The exact comparator is exact for its FP32 scores; that does not make the result equivalent to PyStamps' FP64 arithmetic.

Upstream commit: `257856e3b221140acb4cc4460626b850c0bd4ff1`. The experiment uses its core interpreter, generated-instance JavaScript backend, and [experimental WGSL backend](https://github.com/chenglou/handwired/tree/257856e3b221140acb4cc4460626b850c0bd4ff1/lab/ui-exploration/codegen). No upstream source is modified. A separate, simple 8×8 tiled FP32 matrix shader provides the direct GPU control; it is not a tuned vendor GEMM benchmark.

## Measured search timings

Milliseconds, with inputs and trial weights prepared. GPU timings include each batch's upload, dispatch, readback and winner extraction. NumPy uses the existing Apple Accelerate BLAS. Hardware GPU: Apple, Metal 3, Chrome 155, `isFallbackAdapter=false`. Host: macOS 27.0.1 arm64, Python 3.9.6, NumPy 2.0.2, Bun 1.3.12. Exact CPU model was not queried; environment/configuration and source hashes are in [results.json](results.json).

| Method | 1,024 pixels | 10,000 pixels | 300,000 pixels |
| --- | ---: | ---: | ---: |
| Original per-pixel trial search | 36.724 | 353.795 | 10,676.303 |
| Batched NumPy complex128 (FP64 components) | 0.160 | 1.721 | **58.763** |
| Batched NumPy complex64, FP64 host selection | 0.119 | 1.136 | 39.466 |
| Batched real NumPy FP32, FP64 host selection | 0.138 | 1.269 | 43.699 |
| Handwired CPU interpreter | 8.449 | 81.340 | 2,513.807 |
| Handwired generated JavaScript | 101.360 | 54.514 | 628.425 |
| Handwired strict GPU, host selection | 2.300 | 8.200 | 179.300 |
| Handwired relaxed GPU, host selection | 1.600 | 6.200 | 187.900 |
| Direct tiled GPU matmul, host selection | 1.200 | 6.700 | 195.000 |
| **Handwired GPU with fused selector** | **1.700** | **3.000** | **79.700** |
| Direct tiled GPU with GPU selector | 1.000 | 3.100 | 89.100 |

The fused Handwired implementation is about 134× faster than the current loop, but 1.36× slower than batched FP64 NumPy. It is about 11% faster than this simple direct GPU control at the largest size; that modest result does not establish superiority to optimized GPU matrix libraries. NumPy complex64 was fastest here, but lower precision is not guaranteed to preserve winners on other inputs.

CPU timings are medians of three samples, except the 300,000-pixel original loop, which was measured **once** (not extrapolated). GPU timings are medians of five samples after a first execution. Generated JS was not fully at steady state: its large-case samples were 1,649.6, 628.4 and 612.8 ms, and its first small-case calls were substantially slower. Every sample is retained. Fixed method order and a shared desktop limit performance certainty.

The original-search baseline reproduces trial setup, sums, normalized scores and argmax. It omits the original's unused `C_trial = angle(...)` calculation and does not include refinement. Source-loader tests compare the search-plus-refinement result against actual `ps_topofit` calls.

## Preparation, boundaries and full-function check

- For 300,000 pixels, real FP32 input packing took 35.9 ms; complex64 casting took 5.9 ms; shared denominator preparation took 46.9 ms. These costs are separate from warm search. Fused GPU selectors compare squared magnitudes without normalization and do not need denominators. Correlation-plus-host variants all use original FP64 denominators and FP64 magnitude selection. Source code still computes its own denominator per row.
- Handwired builder/compiler/code-generation setup took about 29 ms, including construction of both correlation and fused programs. It is separate from warm inference. Circuit source bounds are checked by the CPU interpreter; generated/GPU methods consume the already validated fixture.
- GPU batches contain at most 16,384 pixels, including the final partial batch. The host-selector path reads back 66 floats per pixel; fused selection reads back only two. GPU `dispatch_readback` also includes the upstream runner's per-batch shared-buffer upload; this is a small shared-values/guard buffer, not the fixed coefficients embedded in the shader. Direct-control weights upload once at setup.
- `setup_ms` for Handwired includes fetching shader/shared-buffer assets. It is not directly comparable to direct-control setup, whose weights are already loaded. Shader caches affect compilation: first observed strict/fused compilation took around 1.4 seconds, while later cached pipelines took milliseconds. The saved final run retains first fused compilation at 1,390.8 ms. No claim of cold whole-process speedup is made.
- Local HTTP fixture loading, Python-to-browser serialization, page/adapter startup, disk result writes, and final PyStamps refinement are excluded from warm GPU timing. Thus these are **kernel/backend comparisons, not an integrated Python application's latency**. The large-case preparation costs and all transfer boundaries must be considered before integration.
- On an actual 512-row `ps_topofit` sample, the original full function took **38.60 ms**, while batched FP64 search plus the unchanged refinement took **18.72 ms**: about **2.06×**. This smaller improvement illustrates the remaining refinement cost. It is not a measured speedup for all of Step 2 or the whole pipeline.

## Fidelity and the near-tie witness

Every backend's winning trial was compared across every pixel. Every returned correlation component was checked against the source; generated JS and strict GPU correlation outputs were also compared with the actual Handwired interpreter.

For 300,000 pixels:

| Method family | Different winning trials |
| --- | ---: |
| NumPy complex128 | **0** |
| NumPy complex64 with the common FP64 selector | 0 |
| Handwired interpreted/generated/strict/relaxed/fused GPU | **1** |
| Real FP32 NumPy and direct GPU controls | 2 |

No backend changed a winner in the two smaller fixtures. Strict GPU and generated JS matched all Handwired correlation outputs exactly numerically. Fused GPU indices and squared scores matched the FP32 reference derived from the interpreter; 128 rows per case were additionally checked through the actual fused circuit evaluator. These backend agreements do not imply FP64 source equivalence.

The measured synthetic calibration witness at row **207197** selects trial **9** in the original and trial **10** in Handwired. Its original normalized-score gap is only **6.49e-9**. Running the unchanged refinement from those choices changes K by about **0.001163**, C by **0.036925 radians**, and coherence by **0.0001931**. Small score errors therefore cannot be dismissed solely because few winners change.

[near-tie.json](near-tie.json) preserves this compact actual input; the Python and Bun tests reproduce the source choice, Handwired divergence, and resulting phase change. Refinement comparisons cover the first 512 rows plus **every** disagreeing row for each backend. Unchanged winners yield unchanged refinement because that stage consumes original FP64 phase inputs. No full 300,000-row refinement equality claim is made.

FP64 NumPy's maximum correlation error was 2.84e-14 with no changed winners in this dataset. This is strong evidence for further validation, not a proof about all inputs or mathematical ties. A future mixed-precision implementation could detect ambiguous winners and recompute them in FP64, but that fallback is not implemented or benchmarked here.

## Inputs and environment caveats

Fixtures use the same seed/layout as Step 2 random calibration (`RandomState(2005).rand(n_ifg,n_rand).T`), with deterministic synthetic shared baselines, 60 interferograms and `n_trial_wraps=2` (33 trials). They contain finite nonzero unit complex phases. Per-pixel baseline variation and zero-sample masking are outside this shared-weight implementation's domain. All fixture sizes use the same baseline; phase datasets are not prefixes because the original random layout depends on pixel count.

The bundled real-scene archive remains an unavailable Git LFS pointer. Real-scene coherence histograms, candidate thresholds, ground-displacement outputs, memory peaks, and energy were not validated. Three-dimensional unwrapping and other PyStamps stages are outside scope.

This NumPy/Accelerate installation emits spurious divide-by-zero, overflow and invalid warnings during even a 16×16 all-ones multiplication, while returning the exact matrix. This matches [NumPy's reported Accelerate issue](https://github.com/numpy/numpy/issues/28687) and [upstream mitigation](https://github.com/numpy/numpy/pull/30102). Warning messages/counts are retained in results. Finite-output checks, source comparisons and a non-BLAS `einsum` check on 1,024 rows independently validate the values. No production warning suppression or dependency changes were made.

## Reproduce

Use Python with NumPy, Bun, and a Chrome browser with hardware WebGPU. Keep synthetic binary files in a scratch directory; the largest fixture set is approximately 1 GB and is not checked into the repository.

```sh
git clone https://github.com/chenglou/handwired.git /tmp/pystamps-handwired
git -C /tmp/pystamps-handwired checkout 257856e3b221140acb4cc4460626b850c0bd4ff1

# From the PyStamps repository root:
python3 -m unittest experiments.handwired_topofit.test_cpu experiments.handwired_topofit.test_precision
HANDWIRED_DIR=/tmp/pystamps-handwired bun test experiments/handwired_topofit/network.test.js
python3 experiments/handwired_topofit/benchmark.py prepare \
  --root /tmp/pystamps-topofit-data --handwired /tmp/pystamps-handwired
HANDWIRED_DIR=/tmp/pystamps-handwired bun experiments/handwired_topofit/serve.mjs \
  /tmp/pystamps-topofit-data 8770
```

Open `http://127.0.0.1:8770` in Chrome. In its developer console, run sequentially:

```js
const bench = await import('/browser.mjs');
await bench.smokeTest();
await bench.runCase(1024, 5);
await bench.runCase(10000, 5);
await bench.runCase(300000, 5);
```

The local-only server writes GPU outputs into the scratch directory. Then:

```sh
python3 experiments/handwired_topofit/benchmark.py collect \
  --root /tmp/pystamps-topofit-data --handwired /tmp/pystamps-handwired
```

Stop the server when finished. Use a fresh scratch directory to avoid mixing runs. If a GPU method reports an error, retain it and inspect `missing_backend_results`; do not treat an absent backend as a speed or fidelity result.

## Verification and files

Eight Python tests and four Bun tests pass, including zero-mask rejection, coherent phases, variable-amplitude refinement, exact ties, and the measured near-tie witness. Hardware GPU smoke tests cover partial dimensions (3×5×7, 9×9×10, 1×120×66) and the direct winner kernel. All three benchmark sizes ran on a nonfallback Apple GPU. Static Python/JavaScript syntax and whitespace checks were performed; this repository has no configured lint/typecheck suite.

Preparation-time hashes are preserved as recorded. `benchmark.py` received post-measurement collection/reporting edits, and `network.test.js` received the saved-witness test; their delivered hashes therefore differ from those original records. The timed computational kernels were not changed after the final measurements.

- `cpu.py`, `test_cpu.py`, `test_precision.py`: source controls, original-function loader, refinement, numerical tests.
- `network.mjs`, `runtime.mjs`, `run-cpu.mjs`, `network.test.js`: actual upstream circuit, CPU execution, exact-comparison selector and tests.
- `gpu-control.mjs`, `browser.mjs`, `serve.mjs`: direct GPU control, GPU experiments and local fixture/result transport.
- `benchmark.py`, `results.json`, `near-tie.json`: preparation, measurements, comparison evidence and precision witness.

The reusable simplification is precomputing common trial weights and batching the per-pixel correlations. No production refactor was applied. Before production adoption, validate batched FP64 against real calibration histograms and complete Step 2 outputs. Handwired would need a demonstrated advantage over this simpler control and an explicit precision policy.
