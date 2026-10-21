"""Prepare CPU measurements and binary fixtures; collect GPU outputs afterwards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import warnings
import contextlib
import io

import numpy as np
from cpu import fixture, trial_weights, source_search, refine, reference_topofit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def timed(function, repeats, warmup=True):
    if warmup:
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = function()
        samples.append((time.perf_counter() - start) * 1000)
    return output, {'median_ms': statistics.median(samples), 'samples_ms': samples}


def real_search(inputs, weights, denominators, chunk=4096):
    result = np.empty((len(inputs), weights.shape[1]), dtype=np.float32)
    best = np.empty(len(inputs), dtype=np.intp)
    trials = weights.shape[1] // 2
    for start in range(0, len(inputs), chunk):
        stop = min(len(inputs), start + chunk)
        result[start:stop] = inputs[start:stop] @ weights
        values = result[start:stop].astype(np.float64)
        scores = np.hypot(values[:, :trials], values[:, trials:]) / denominators[start:stop, None]
        best[start:stop] = scores.argmax(axis=1)
    return result, best


def complex_search(phase, weights, denominators, chunk=4096):
    """Same FP64 host selector and prevalidated-input boundary as other controls."""
    sums = np.empty((len(phase), weights.shape[1]), dtype=np.result_type(phase, weights))
    best = np.empty(len(phase), dtype=np.intp)
    for start in range(0, len(phase), chunk):
        stop = min(start+chunk, len(phase))
        sums[start:stop] = phase[start:stop] @ weights
        values = sums[start:stop].astype(np.complex128)
        scores = np.hypot(values.real, values.imag) / denominators[start:stop, None]
        best[start:stop] = scores.argmax(axis=1)
    return sums, best


def compare(sums, best, reference_sums, reference_best, denominators):
    if not np.all(np.isfinite(sums)) or not np.all(np.isfinite(reference_sums)):
        raise ValueError('nonfinite correlation sums')
    difference = np.abs(sums - reference_sums)
    flips = np.flatnonzero(best != reference_best)
    reference_scores = np.abs(reference_sums) / denominators[:, None]
    scores = np.hypot(sums.real.astype(np.float64), sums.imag.astype(np.float64)) / denominators[:,None]
    top = np.partition(reference_scores, -2, axis=1)[:, -2:]
    margins = top[:,1]-top[:,0]
    return {'max_abs_sum_error': float(difference.max(initial=0)),
            'rms_sum_error': float(np.sqrt(np.mean(difference**2))),
            'max_abs_coherence_score_error':float(np.max(np.abs(scores-reference_scores))),
            'minimum_reference_winner_margin':float(margins.min()),
            'disagreement_reference_margins':margins[flips].tolist(),
            'winner_disagreements': len(flips), 'disagreement_indices': flips.tolist()}


def prepare_case(count, root, upstream, repeats):
    directory = root / str(count)
    directory.mkdir(parents=True, exist_ok=True)
    phase, bperp, wraps = fixture(count)
    (weights, weight_time) = timed(lambda: trial_weights(bperp, wraps), repeats)
    start = time.perf_counter()
    inputs = np.concatenate((phase.real, phase.imag), axis=1).astype('<f4')
    dense = np.block([[weights.real, weights.imag], [-weights.imag, weights.real]]).astype('<f4')
    real_prep_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    phase32, weights32 = phase.astype(np.complex64), weights.astype(np.complex64)
    complex_prep_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    denominators = np.abs(phase).sum(axis=1)
    denominator_ms = (time.perf_counter()-start)*1000
    metadata = {'count': count, 'ifgs': phase.shape[1], 'trials': weights.shape[1],
                'inputs': dense.shape[0], 'outputs': dense.shape[1], 'wraps': wraps, 'seed': 2005}
    (directory / 'input.json').write_text(json.dumps(metadata))
    inputs.tofile(directory / 'input.bin')
    dense.tofile(directory / 'weights.bin')
    denominators.astype('<f8').tofile(directory / 'denominators.bin')
    np.save(directory / 'phase.npy', phase)
    np.save(directory / 'baseline.npy', bperp)
    # Large source loops are measured once, explicitly marked, not extrapolated.
    source_repeats = 1 if count >= 100000 else repeats
    source_search(phase[:min(128, count)], bperp, wraps)
    (expected, best), source_time = timed(lambda: source_search(phase, bperp, wraps), source_repeats, warmup=False)
    np.save(directory / 'reference-sums.npy', expected)
    np.save(directory / 'reference-best.npy', best)
    report = {'shape': metadata, 'trial_weight_preparation': weight_time, 'source_search': source_time, 'methods': {}}
    config=io.StringIO()
    with contextlib.redirect_stdout(config): np.show_config()
    report['environment']={'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'blas_config':config.getvalue()}
    report['experiment_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.mjs','.js')}
    report['input_preparation']={'real32_packing_ms':real_prep_ms,'complex64_cast_ms':complex_prep_ms,'shared_denominators_ms':denominator_ms}
    # Independent non-BLAS check for this environment's spurious FP flags.
    check_count = min(count, 1024)
    independent = np.einsum('ni,it->nt', phase[:check_count], weights, optimize=False)
    np.testing.assert_allclose(independent, expected[:check_count], rtol=1e-12, atol=1e-12)
    report['independent_einsum_check'] = {'count':check_count,'max_abs_error':float(np.max(np.abs(independent-expected[:check_count])))}
    for name, fn in [
        ('numpy_complex128', lambda: complex_search(phase, weights, denominators)),
        ('numpy_complex64', lambda: complex_search(phase32, weights32, denominators)),
        ('numpy_real32', lambda: real_search(inputs, dense, denominators)),
    ]:
        (actual, selected), timing = timed(fn, repeats)
        if name == 'numpy_real32':
            actual = actual[:, :metadata['trials']].astype(np.float64) + 1j * actual[:, metadata['trials']:]
        np.save(directory / (name + '-best.npy'), selected)
        report['methods'][name] = {'search': timing, 'correctness': compare(actual, selected, expected, best, denominators)}
    sample = min(512, count)
    reference = [reference_topofit(row, bperp, wraps) for row in phase[:sample]]
    refined = refine(phase[:sample], bperp, wraps, best[:sample])
    for name, index in [('K', 0), ('C', 1), ('coh', 2)]:
        values = np.array([np.asarray(row[index]).item() for row in reference])
        np.testing.assert_allclose(refined[name], values, rtol=0, atol=0)
    _, full_timing = timed(lambda: [reference_topofit(row, bperp, wraps) for row in phase[:sample]], repeats)
    report['actual_full_topofit_sample'] = {'count': sample, 'timing': full_timing, 'refinement_exact': True}
    _, batched_full_time = timed(lambda: refine(phase[:sample],bperp,wraps,complex_search(phase[:sample],weights,denominators[:sample])[1]),repeats)
    report['actual_full_topofit_sample']['batched_search_plus_unchanged_refinement'] = batched_full_time
    start = time.perf_counter()
    subprocess.run(['bun', str(HERE / 'run-cpu.mjs'), str(directory), str(repeats)],
                   env=dict(os.environ, HANDWIRED_DIR=str(upstream)), check=True, stdout=subprocess.DEVNULL)
    report['handwired_process_all_runs_ms'] = (time.perf_counter() - start) * 1000
    report['handwired_cpu'] = json.loads((directory / 'cpu-handwired.json').read_text())
    (directory / 'cpu.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'count': count, 'source_ms': source_time['median_ms'],
                      'numpy128_ms': report['methods']['numpy_complex128']['search']['median_ms'],
                      'handwired': report['handwired_cpu']['methods']}), flush=True)


def collect_case(directory):
    report = json.loads((directory / 'cpu.json').read_text())
    metadata = report['shape']
    count, trials = metadata['count'], metadata['trials']
    phase, bperp = np.load(directory / 'phase.npy'), np.load(directory / 'baseline.npy')
    expected, best = np.load(directory / 'reference-sums.npy'), np.load(directory / 'reference-best.npy')
    denominators = np.fromfile(directory / 'denominators.bin','<f8')
    variants = list(report['methods'])
    report['missing_backend_results']=[]
    for name in ['interpreted', 'generated', 'gpu-strict', 'gpu-relaxed', 'gpu-matmul']:
        path = directory / (name + '.bin')
        if not path.exists():
            report['missing_backend_results'].append(name)
            continue
        data = np.fromfile(path, '<f4').reshape(count, 2 * trials)
        actual = data[:, :trials].astype(np.float64) + 1j * data[:, trials:]
        selected = np.fromfile(directory / (name + '-best.bin'), '<u4')
        report['methods'][name] = {'correctness': compare(actual, selected, expected, best,denominators)}
        if name != 'interpreted':
            interpreted = np.fromfile(directory / 'interpreted.bin', '<f4').reshape(count, 2*trials)
            report['methods'][name]['handwired_interpreter_comparison'] = {
                'unequal_components':int(np.count_nonzero(data != interpreted)),
                'max_abs_error':float(np.max(np.abs(data.astype(np.float64)-interpreted)))}
        np.save(directory / (name + '-best.npy'), selected)
        variants.append(name)
    for name in ['gpu-fused','gpu-matmul-winner']:
        path=directory/(name+'.bin')
        if not path.exists():
            report['missing_backend_results'].append(name)
            continue
        values=np.fromfile(path,'<f4').reshape(count,2)
        selected=np.fromfile(directory/(name+'-best.bin'),'<u4')
        if not np.all(np.isfinite(values)) or np.any(selected>=trials): raise ValueError('invalid fused results')
        flips=np.flatnonzero(selected!=best)
        reference_scores=np.abs(expected)/denominators[:,None]
        top=np.partition(reference_scores,-2,axis=1)[:,-2:]
        margins=top[:,1]-top[:,0]
        reference_fused=np.fromfile(directory/'fused-reference.bin','<f4').reshape(count,2)
        report['methods'][name]={'correctness':{'winner_disagreements':len(flips),'disagreement_indices':flips.tolist(),
          'disagreement_reference_margins':margins[flips].tolist(),
          'maximum_selected_coherence_loss':float(np.max(reference_scores[np.arange(count),best]-reference_scores[np.arange(count),selected]))},
          'handwired_fused_interpreter_comparison':{'winner_disagreements':int(np.count_nonzero(values[:,0]!=reference_fused[:,0])),
          'max_squared_score_error':float(np.max(np.abs(values[:,1]-reference_fused[:,1])))}}
        np.save(directory/(name+'-best.npy'),selected)
        variants.append(name)
    for name in variants:
        selected = np.load(directory / (name + '-best.npy'))
        flips = np.flatnonzero(selected != best)
        sample = np.unique(np.concatenate((np.arange(min(512, count)), flips)))
        baseline = refine(phase[sample], bperp, metadata['wraps'], best[sample])
        candidate = refine(phase[sample], bperp, metadata['wraps'], selected[sample])
        report['methods'][name]['refinement'] = {'sample_count': len(sample),
          'includes_all_winner_disagreements': True,
          **{key + '_max_abs_error': float(np.max(np.abs(candidate[key] - baseline[key]))) for key in ('K','C','coh','phase_residual')}}
    gpu_path = directory / 'gpu.json'
    if gpu_path.exists():
        report['gpu'] = json.loads(gpu_path.read_text())
    if (directory/'gpu.smoke.json').exists():
        report['gpu_smoke_tests']=json.loads((directory/'gpu.smoke.json').read_text())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'collect'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--handwired', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[1024, 10000, 300000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=HERE/'results.json')
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.counts):
        parser.error('positive counts and repeats required')
    if args.mode == 'prepare':
        for count in args.counts:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always', RuntimeWarning)
                prepare_case(count, args.root, args.handwired, args.repeats)
            warning_counts = {}
            for warning in recorded:
                key = str(warning.message)
                warning_counts[key] = warning_counts.get(key, 0) + 1
            path = args.root / str(count) / 'cpu.json'
            report = json.loads(path.read_text())
            report['runtime_warnings'] = warning_counts
            path.write_text(json.dumps(report, indent=2))
    else:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always',RuntimeWarning)
            cases=[collect_case(args.root / str(count)) for count in args.counts]
        report = {'platform': platform.platform(), 'numpy': np.__version__,
          'pystamps_commit': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
          'handwired_commit': subprocess.check_output(['git','-C',str(args.handwired),'rev-parse','HEAD'],text=True).strip(),
          'source_sha256': hashlib.sha256((ROOT/'src/step/ps_topofit.py').read_bytes()).hexdigest(),
          'collection_warning_messages':sorted(set(str(w.message) for w in recorded)),
          'cases': cases}
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print('Saved', args.output)


if __name__ == '__main__':
    main()
