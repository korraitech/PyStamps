# Handwired feasibility: Step 7 neighbour smoothing

**Decision:** Step 7's neighbour extrema are a good first compatibility experiment for Handwired, but the measured CPU network is not the best performance option. A NumPy scatter implementation is about 27 times faster than this Handwired implementation. Keep Handwired experimental and validate the NumPy approach on real scenes before integrating it.

## Why this step

`smooth_scla` in `src/step/step_7_ps_scla.py:39-61` computes neighbour minimum/maximum values for K and C, then clips outliers. The current implementation performs small NumPy operations inside a Python loop over every edge. These operations map naturally to Handwired's exact min/max circuits.

Step 2 is a stronger candidate for a later overall runtime improvement: `src/step/step_2_ps_estmgm.py:27,116-118` performs 300,000 topographic fits during random calibration, then fits every candidate during estimation. Its `ps_topofit` kernel involves complex exponentials, phase angles, argmax and least squares. Precomputing shared trial matrices and batching would be a more direct first approach there. This ranking comes from code inspection, not whole-pipeline profiling.

## What actually ran

This experiment imports **the real upstream Handwired runtime**, pinned to commit `257856e3b221140acb4cc4460626b850c0bd4ff1`. No trained model or imitation runtime is used. Handwired requires manually written builder code; it does not automatically convert Python or JavaScript source. Its general runtime is CPU-based; GPU execution is demo-specific. See [upstream documentation](https://github.com/chenglou/handwired/blob/257856e3b221140acb4cc4460626b850c0bd4ff1/docs/README.md).

One circuit per neighbour count computes both bounds using `emitMinimumExact` and `emitMaximumExact`. A degree-six circuit has 180 live operations across 80 layers. Plain arithmetic min/max macros can lose precision on fractional inputs, so this experiment uses the exact variants. See [upstream expression contracts](https://github.com/chenglou/handwired/blob/257856e3b221140acb4cc4460626b850c0bd4ff1/docs/expressions.md).

The result is a **hybrid bounds network plus host-side gathering and clipping**, not a neural implementation of the entire step. Original neighbour bounds are FP32; K/C inputs can be FP64. Keeping final comparisons and unclipped focal values in FP64 preserves the original numerical outputs. Host gathering, validation, output allocation and clipping are included in warm network timings. Circuit construction and adjacency setup are measured separately.

Four controls use identical inputs:

1. The original reduction/clamp statements, extracted directly from the source AST, with logging suppressed and input copies to avoid mutation between runs.
2. NumPy `minimum.at` / `maximum.at`, retaining FP32 bounds and the original comparison/assignment order.
3. A plain JavaScript edge loop with FP32 bound arrays and FP64 outputs.
4. The upstream Handwired circuits with host FP64 comparisons.

## Results

Measured on macOS 27.0.1 arm64, Python 3.9.6, NumPy 2.0.2 and Bun 1.3.12. PyStamps commit: `4683ca20af0e0db6fc84d35bf2a6a4b0501765b1`. These are local synthetic measurements, not production service performance estimates.

Warm median milliseconds, nine repetitions after warmup, FP64 input:

| Points | Edges | Original Python | Handwired hybrid | NumPy scatter | Plain JS | Handwired cold round trip |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2,945 | 20.194 | 2.627 | 0.099 | 0.034 | 32.012 |
| 10,000 | 29,601 | 198.873 | 25.395 | 0.888 | 0.343 | 65.113 |
| 40,000 | 119,201 | 795.708 | 102.798 | 3.747 | 1.599 | 165.262 |

At 40,000 points, Handwired is **7.74× faster than the original loop**, but **27.44× slower than NumPy** and **64.28× slower than plain JS**. NumPy is **212.39× faster than the original loop** for this isolated kernel. The comparable FP32 case gives 6.77× for Handwired and 180.12× for NumPy versus the original.

Network compilation costs about 4–5 ms for these four distinct degrees; adjacency setup takes about 0.5–3.6 ms. Cold round trips are medians of three fresh Bun processes, including binary input writing, runtime import, graph construction, one evaluation and output reading. They exclude the native JS control. At the smallest size, cold Handwired is slower than the existing in-process Python kernel.

All six size/precision cases matched the original numerical outputs exactly: maximum absolute error **0**, with no tolerance relaxation. Signed-zero bit patterns are not compared. The NumPy and plain JS controls also matched. [results.json](results.json) contains every timing sample, setup cost, source hash and version. [results.initial.json](results.initial.json) retains the first seven-repeat run; a noisy 40,000-point FP32 scatter result motivated the second run after verification jobs finished. The conclusion holds in both runs.

## Scope and limitations

- Seeded synthetic triangular grids, degrees 2, 3, 4 and 6; they are not real Delaunay triangulations or measured radar scenes. Tests additionally exercise degree 1 and 11.
- Handwired accepts only finite values within ±16,384 and nonisolated vertices here. Nonfinite values and isolated vertices are rejected, not silently approximated. NumPy tests separately retain the original NaN and isolated-vertex behaviour.
- No Delaunay construction, HDF5 I/O, SCLA fitting, phase reconstruction, full Step 7 or end-to-end pipeline timing is included. No GPU result is claimed.
- Warm timings assume reusable circuits/topology. Workload order is fixed; repeated local timings are not a controlled hardware study. Resident memory and energy were not measured.
- `test/export.tar.gz` is a Git LFS pointer for a 339,886,672-byte dataset, not an available archive. Real-scene scientific equivalence and end-to-end speedups remain untested.
- Production code and dependencies are unchanged. This experiment demonstrates that removing the Python edge loop is valuable; it does not establish a benefit specific to neural execution, nor rule out a future specialized GPU implementation.

## Reproduce

Use an existing Python environment with NumPy and an installed Bun. No Handwired package installation is required for its standalone core runtime.

```sh
git clone https://github.com/chenglou/handwired.git /tmp/pystamps-handwired
git -C /tmp/pystamps-handwired checkout 257856e3b221140acb4cc4460626b850c0bd4ff1

# From the PyStamps repository root:
python3 -m unittest discover -s experiments/handwired_scla -p 'test_*.py'
HANDWIRED_DIR=/tmp/pystamps-handwired bun test experiments/handwired_scla/runner.test.js
python3 experiments/handwired_scla/benchmark.py --handwired /tmp/pystamps-handwired --repeats 9
```

Validation performed: 3 Python tests, 4 Bun tests, all six benchmark equality checks, Python byte compilation, JavaScript syntax checks, and 174 upstream runtime tests. There is no repository lint/typecheck configuration; no full pipeline validation was performed.

Files added: `benchmark.py` (reference, NumPy control, fixtures and orchestration), `runner.mjs` (actual circuits and JS control), two test files, this report, and raw results. No production simplifications were applied. The experimental simplification is replacing per-edge Python operations with existing NumPy scatter reductions; no new abstraction or dependency is needed in production to pursue it.

Recommended next work: validate the NumPy reduction on real Step 7 intermediates, then profile Step 2's repeated `ps_topofit` work. Treat Handwired GPU exploration as a separate backend project rather than a drop-in optimization.
