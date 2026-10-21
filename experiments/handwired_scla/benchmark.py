"""Isolated, reproducible Step 7 experiment. Requires NumPy and Bun, no new production dependencies."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'src/step/step_7_ps_scla.py'


def load_source_kernel():
    """Compile the actual reduction/clamp statements, without HDF5/SciPy imports."""
    tree = ast.parse(SOURCE.read_text())
    body = next(node.body for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'smooth_scla')
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'Kneigh_min')
    end = next(i for i, node in enumerate(body) if i > start and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'bp')
    wrapper = ast.parse('def source_kernel(edgs, k, c):\n n_ps = len(k)\n n_edge = len(edgs)\n K_ps_uw = k.copy()\n C_ps_uw = c.copy()\n')
    wrapper.body[0].body.extend(body[start:end])
    wrapper.body[0].body.extend(ast.parse('return K_ps_uw, C_ps_uw').body)
    scope = {'np': np, 'print': lambda *args: None}
    exec(compile(ast.fix_missing_locations(wrapper), str(SOURCE), 'exec'), scope)
    return scope['source_kernel']


source_kernel = load_source_kernel()


def numpy_kernel(edges, k, c):
    """Scatter control; retain FP32 bounds and both masks before assignment."""
    a, b = edges.T
    result = []
    for values in (k, c):
        lower = np.full(len(values), np.inf, dtype=np.float32)
        upper = np.full(len(values), -np.inf, dtype=np.float32)
        rounded = values.astype(np.float32)
        np.minimum.at(lower, a, rounded[b])
        np.minimum.at(lower, b, rounded[a])
        np.maximum.at(upper, a, rounded[b])
        np.maximum.at(upper, b, rounded[a])
        output = values.copy()
        above, below = values > upper, values < lower
        output[above] = upper[above]
        output[below] = lower[below]
        result.append(output)
    return tuple(result)


def fixture(side, dtype=np.float64):
    """Triangular grid, not a measured scene or a Delaunay benchmark."""
    grid = np.arange(side * side, dtype=np.uint32).reshape(side, side)
    edges = np.concatenate([
        np.column_stack((grid[:, :-1].ravel(), grid[:, 1:].ravel())),
        np.column_stack((grid[:-1, :].ravel(), grid[1:, :].ravel())),
        np.column_stack((grid[:-1, :-1].ravel(), grid[1:, 1:].ravel())),
    ])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rng = np.random.default_rng(20261021 + side)
    k = rng.normal(0, 0.02, side * side).astype(dtype)
    c = rng.normal(0, 3.0, side * side).astype(dtype)
    k[::101] *= 100
    c[::101] *= 100
    k[::17] = 0
    c[::11] = 0.3
    return edges, k, c


def timed(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = fn()
        samples.append((time.perf_counter() - start) * 1000)
    return output, {'median_ms': statistics.median(samples), 'samples_ms': samples}


def run_case(side, upstream, repeats, dtype=np.float64):
    edges, k, c = fixture(side, dtype)
    expected, original_time = timed(lambda: source_kernel(edges, k, c), repeats)
    scatter, scatter_time = timed(lambda: numpy_kernel(edges, k, c), repeats)
    expected = np.concatenate(expected)
    np.testing.assert_array_equal(np.concatenate(scatter), expected)
    with tempfile.TemporaryDirectory(prefix='pystamps-handwired-') as directory:
        directory = Path(directory)
        (directory / 'input.json').write_text(json.dumps({'n': len(k)}))
        edges.astype('<u4').tofile(directory / 'edges.bin')
        np.concatenate((k, c)).astype('<f8').tofile(directory / 'values.bin')
        env = dict(os.environ, HANDWIRED_DIR=str(upstream))
        command = ['bun', str(HERE / 'runner.mjs'), str(directory)]
        result = subprocess.run(command + [str(repeats), '3'], env=env, capture_output=True, text=True, check=True)
        js = json.loads(result.stdout)
        checks = {}
        for name in ('network', 'native'):
            actual = np.fromfile(directory / (name + '.bin'), dtype='<f8')
            np.testing.assert_array_equal(actual, expected)
            checks[name] = {'equal_to_source': True, 'max_abs_error': float(np.max(np.abs(actual - expected)))}
        # A fresh Bun process: serialize input, build graph, execute once, read output.
        cold_samples = []
        for _ in range(3):
            start = time.perf_counter()
            edges.astype('<u4').tofile(directory / 'edges.bin')
            np.concatenate((k, c)).astype('<f8').tofile(directory / 'values.bin')
            subprocess.run(command + ['1', '0'], env=env, capture_output=True, text=True, check=True)
            cold = np.fromfile(directory / 'network.bin', dtype='<f8')
            cold_samples.append((time.perf_counter() - start) * 1000)
            np.testing.assert_array_equal(cold, expected)
    return {
        'points': len(k), 'edges': len(edges), 'dtype': np.dtype(dtype).name,
        'source': original_time, 'numpy_scatter': scatter_time,
        'handwired': js,
        'cold_roundtrip': {'median_ms': statistics.median(cold_samples), 'samples_ms': cold_samples},
        'correctness': checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handwired', type=Path, required=True)
    parser.add_argument('--sides', type=int, nargs='+', default=[32, 100, 200])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, default=HERE / 'results.json')
    args = parser.parse_args()
    if args.repeats < 2 or any(side < 2 for side in args.sides):
        parser.error('use at least two repeats and sides >= 2')
    upstream = args.handwired.resolve()
    report = {
        'platform': platform.platform(), 'machine': platform.machine(),
        'python': sys.version, 'numpy': np.__version__,
        'pystamps_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'handwired_commit': subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'Finite synthetic triangular meshes; bounds network + host FP64 clamps; CPU only; excludes Delaunay, HDF5 and remainder of Step 7.',
        'cases': [],
    }
    for dtype in (np.float32, np.float64):
        for side in args.sides:
            case = run_case(side, upstream, args.repeats, dtype)
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
