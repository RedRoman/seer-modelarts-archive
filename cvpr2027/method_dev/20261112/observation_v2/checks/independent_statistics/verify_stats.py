"""Independent JSON+NumPy check of observation-v2 statistics; no project imports.

This verifier reconstructs effects directly from each of the 31 raw methods.
A single RNG0 10000x8 draw matrix is converted to scene multiplicities, then
multiplied by scene means. Three head seeds are averaged before scene sampling.
This is independent math verification, not a substitute for provenance audit.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

QUALITY = ('psnr_rgb', 'psnr_y', 'lpips', 'tlp_gap')
METRICS = (*QUALITY, 'tlp', 'gt_tlp')
CANDIDATES = ('raw_detail', 'same_frame_detail', 'mismatched_detail')
FIXED = ('base', *(f'{c}/{f}' for c in CANDIDATES for f in ('initial_thirds', 'fixed_average')))
LEARNED = ('raw_node', 'same_frame_node', 'mismatched_node', 'trajectory',
           'trajectory_fixed_edge', 'trajectory_no_cross', 'trajectory_scrambled_cross', 'v1_raw_direct')
LABELS = (*FIXED, *LEARNED)
METHODS = {*FIXED, *(f'{label}_s{seed}' for label in LEARNED for seed in range(3))}
PAIRS = (
    ('raw_node', 'same_frame_node'), ('raw_node', 'mismatched_node'), ('raw_node', 'base'),
    ('raw_node', 'raw_detail/initial_thirds'), ('raw_node', 'v1_raw_direct'),
    ('trajectory', 'raw_node'), ('trajectory', 'trajectory_fixed_edge'),
    ('trajectory', 'trajectory_no_cross'), ('trajectory', 'trajectory_scrambled_cross'),
    ('trajectory', 'base'), ('trajectory', 'raw_detail/initial_thirds'), ('trajectory', 'v1_raw_direct'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def scene(name):
    match = re.fullmatch(r'BVI/([^/]+)__(10|20)fps', name)
    require(match is not None, f'Unexpected BVI name: {name}')
    return match.group(1)  # Keep the entire scene identifier, including any _2.


def rows_by_name(raw):
    rows = raw['per_sequence']
    by_name = {r['name']: r for r in rows}
    require(len(rows) == len(by_name) == 16, 'Need 16 distinct raw sequence names')
    names = sorted(by_name)
    scenes = sorted({scene(n) for n in names})
    require(len(scenes) == 8, 'Need eight complete scenes')
    for s in scenes:
        require({n.split('__')[-1] for n in names if scene(n) == s} == {'10fps', '20fps'},
                f'Missing/duplicate condition: {s}')
    return names, scenes, by_name


class Checks:
    def __init__(self):
        self.categories = {}
        self.failures = []

    def compare(self, category, location, observed, expected, tolerance=1e-12):
        a, b = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
        require(a.shape == b.shape, f'Shape mismatch at {location}: {a.shape} vs {b.shape}')
        require(np.isfinite(a).all() and np.isfinite(b).all(), f'Nonfinite statistic at {location}')
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        c = self.categories.setdefault(category, {'scalar_checks': 0, 'max_abs_error': 0.0,
                                                  'max_error_location': None, 'tolerance': tolerance})
        c['scalar_checks'] += int(a.size)
        require(c['tolerance'] == tolerance, f'Inconsistent tolerance for {category}')
        if error > c['max_abs_error']:
            c['max_abs_error'], c['max_error_location'] = error, location
        if error > tolerance:
            self.failures.append({'location': location, 'category': category, 'max_abs_error': error,
                                  'tolerance': tolerance})


def raw_arrays(raw, arm, checks):
    require(raw['domain'] == 'bvi' and raw['split'] == 'val' and raw['n'] == 16, 'Wrong target scope')
    names, scenes, rows = rows_by_name(raw)
    require(set(raw['means']) == METHODS and all(set(r['methods']) == METHODS for r in rows.values()),
            'Raw 31-method coverage differs')
    values = {}
    for method in sorted(METHODS):
        require(set(raw['means'][method]) == set(METRICS), f'Mean metric schema: {method}')
        values[method] = {}
        for key in METRICS:
            require(all(set(rows[n]['methods'][method]) == set(METRICS) for n in names), 'Row metric schema differs')
            v = np.asarray([rows[n]['methods'][method][key] for n in names], dtype=np.float64)
            require(v.shape == (16,) and np.isfinite(v).all(), 'Raw values must be finite vectors')
            checks.compare('raw_reported_means', f'{arm}/{method}/{key}', v.sum() / 16,
                           raw['means'][method][key], 1e-10)
            values[method][key] = v
        checks.compare('raw_temporal_gap', f'{arm}/{method}/gap', values[method]['tlp_gap'],
                       np.abs(values[method]['tlp'] - values[method]['gt_tlp']))
    for method in sorted(METHODS):
        checks.compare('within_arm_gt', f'{arm}/{method}/gt', values[method]['gt_tlp'],
                       values['base']['gt_tlp'], 0.0)
    return {'names': names, 'scenes': scenes, 'values': values}


def matrix(arm, label, metric):
    if label in FIXED:
        return np.tile(arm['values'][label][metric][None, :], (3, 1))
    return np.asarray([arm['values'][f'{label}_s{s}'][metric] for s in range(3)])


def scene_indices(names, scenes):
    return [np.asarray([i for i, n in enumerate(names) if scene(n) == s]) for s in scenes]


def paired(effects, indices, multiplicities, metric):
    require(effects.shape == (3, 16), 'Expected paired head-seed x sequence effects')
    sequence_values = effects.sum(axis=0) / 3.0
    scene_values = np.asarray([sequence_values[idx].sum() / len(idx) for idx in indices])
    boot = multiplicities @ scene_values / 8.0
    interval = np.percentile(boot, [2.5, 97.5])
    sign = 1 if metric.startswith('psnr') else -1
    seeds = effects.sum(axis=1) / 16.0
    return {'difference': float(sequence_values.sum() / 16.0), 'scene_ci95': interval.tolist(),
            'per_selector_seed_difference': seeds.tolist(), 'selector_seeds_better': int((sign * seeds > 0).sum()),
            'sequence_values': sequence_values.tolist(), 'scene_values': scene_values.tolist(),
            'per_seed_sequence_values': effects.tolist(),
            'per_seed_scene_values': np.asarray([effects[:, idx].sum(axis=1) / len(idx) for idx in indices]).T.tolist(),
            'benefit_sign': sign}


def check_pair(computed, published, path, names, scenes, checks, interval_category):
    checks.compare('contrast_differences', path + '/difference', computed['difference'], published['difference'])
    checks.compare(interval_category, path + '/scene_ci95', computed['scene_ci95'], published['scene_ci95'])
    checks.compare('contrast_per_seed', path + '/seeds', computed['per_selector_seed_difference'], published['per_selector_seed_difference'])
    checks.compare('favorable_seed_counts', path + '/better', computed['selector_seeds_better'], published['selector_seeds_better'], 0.0)
    checks.compare('contrast_sequence_values', path + '/sequence', computed['sequence_values'],
                   [published['seed_averaged_sequence_differences'][n] for n in names])
    checks.compare('contrast_scene_values', path + '/scenes', computed['scene_values'],
                   [published['seed_averaged_scene_differences'][s] for s in scenes])
    checks.compare('contrast_per_seed_sequence', path + '/seedsequence', computed['per_seed_sequence_values'],
                   [[published['per_selector_seed_sequence_differences'][str(seed)][n] for n in names] for seed in range(3)])
    checks.compare('contrast_per_seed_scene', path + '/seedscene', computed['per_seed_scene_values'],
                   [[published['per_selector_seed_scene_differences'][str(seed)][s] for s in scenes] for seed in range(3)])
    benefit = published['benefit_oriented']
    sign = computed['benefit_sign']
    for key, value in [('difference', computed['difference']),
                       ('per_selector_seed_difference', computed['per_selector_seed_difference'])]:
        checks.compare('benefit_direction', path + '/benefit/' + key, np.asarray(value) * sign, benefit[key])
    ci = np.asarray(computed['scene_ci95'])
    checks.compare('benefit_direction', path + '/benefit/ci', ci if sign > 0 else -ci[::-1], benefit['scene_ci95'])
    checks.compare('benefit_direction', path + '/benefit/sign', sign, benefit['sign_applied_to_raw_difference'])
    for key, labels, vector in [('seed_averaged_sequence_differences', names, computed['sequence_values']),
                               ('seed_averaged_scene_differences', scenes, computed['scene_values'])]:
        checks.compare('benefit_direction', path + '/benefit/' + key, sign * np.asarray(vector),
                       [benefit[key][name] for name in labels])


def drift(new, old, label):
    names, _, rows = rows_by_name(old)
    require(names == new['names'], f'Legacy target names differ: {label}')
    methods = {'base': 'base', 'raw_detail/initial_thirds': 'raw_detail/initial_thirds',
               'raw_detail/fixed_average': 'raw_detail/fixed_average',
               **{f'v1_raw_direct_s{s}': f'raw_detail_s{s}/direct_mse' for s in range(3)}}
    result = {}
    for current_method, previous_method in methods.items():
        result[current_method] = {}
        for key in METRICS:
            previous = np.asarray([rows[n]['methods'][previous_method][key] for n in names], dtype=np.float64)
            difference = new['values'][current_method][key] - previous
            result[current_method][key] = {
                'old_method': previous_method, 'old_mean': float(previous.mean()),
                'new_mean': float(new['values'][current_method][key].mean()),
                'mean_signed_difference': float(difference.mean()), 'mean_abs_difference': float(abs(difference).mean()),
                'max_abs_difference': float(abs(difference).max()),
                'per_sequence_difference': dict(zip(names, difference.tolist())),
            }
    return {'purpose': 'numerical replay drift, no zero-equality requirement; not a head-training effect',
            'method_count': len(methods), 'metrics': result}


def verify(args):
    paths = {'pretrained': args.pretrained, 'none': args.none, 'main_analysis': args.main_analysis}
    original = {arm: load(paths[arm]) for arm in ('pretrained', 'none')}
    primary = load(args.main_analysis)
    checks = Checks()
    arms = {arm: raw_arrays(original[arm], arm, checks) for arm in ('pretrained', 'none')}
    p, n = arms['pretrained'], arms['none']
    require(p['names'] == n['names'] and p['scenes'] == n['scenes'], 'Target arms are not paired')
    names, scenes = p['names'], p['scenes']
    checks.compare('between_arm_gt', 'P/N base gt', p['values']['base']['gt_tlp'], n['values']['base']['gt_tlp'], 1e-8)
    indices = scene_indices(names, scenes)
    draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
    counts = np.stack([np.bincount(row, minlength=8) for row in draws])
    require(np.all(counts.sum(1) == 8), 'Invalid scene multiplicity draws')
    recomputed = {'within_arms': {}, 'between_arms': {}}
    expected_contrasts = {left + '_minus_' + right for left, right in PAIRS}
    require(set(primary['within_arms']) == {'pretrained', 'none'}, 'Main arms differ')
    require(primary['all_methods'] == sorted(METHODS), 'Main method list differs')
    for arm, data in arms.items():
        ref = primary['within_arms'][arm]
        require(set(ref['individual_methods']) == METHODS and set(ref['absolute']) == set(LABELS), 'Main absolute coverage differs')
        absolute = {}
        for method in sorted(METHODS):
            for key in METRICS:
                v = data['values'][method][key]
                q = ref['individual_methods'][method][key]
                checks.compare('individual_analysis_means', f'{arm}/{method}/{key}', v.mean(), q['mean'])
                checks.compare('individual_analysis_sequence', f'{arm}/{method}/{key}/seq', v, [q['per_sequence'][x] for x in names])
                checks.compare('individual_analysis_scene', f'{arm}/{method}/{key}/scene',
                               [v[idx].mean() for idx in indices], [q['per_scene'][x] for x in scenes])
        for label in LABELS:
            absolute[label] = {}
            for key in METRICS:
                a = matrix(data, label, key)
                per_seed = a.sum(1) / 16.0
                absolute[label][key] = {'mean': float(a.mean()), 'per_selector_seed': per_seed.tolist()}
                q = ref['absolute'][label][key]
                checks.compare('grouped_absolute_means', f'{arm}/{label}/{key}', a.mean(), q['mean'])
                checks.compare('grouped_per_seed_means', f'{arm}/{label}/{key}/seeds', per_seed, q['per_selector_seed'])
                checks.compare('grouped_seed_range', f'{arm}/{label}/{key}/range', [per_seed.min(), per_seed.max()], q['seed_range'])
                checks.compare('grouped_scene_means', f'{arm}/{label}/{key}/scenes',
                               [a[:, idx].mean() for idx in indices], [q['scene_means'][x] for x in scenes])
                require(q['fixed_control_repeated_for_pairing_only'] == (label in FIXED), 'Fixed-control semantics differ')
        require(set(ref['comparisons']) == expected_contrasts, 'Within 12 contrasts differ')
        comparisons = {}
        for left, right in PAIRS:
            name = left + '_minus_' + right
            comparisons[name] = {}
            for key in QUALITY:
                r = paired(matrix(data, left, key) - matrix(data, right, key), indices, counts, key)
                comparisons[name][key] = r
                check_pair(r, ref['comparisons'][name][key], f'{arm}/{name}/{key}', names, scenes, checks, 'within_ci_endpoints')
        recomputed['within_arms'][arm] = {'absolute': absolute, 'comparisons': comparisons}
    expected_between = {'absolute_pretrained_minus_none/' + label for label in LABELS} | {
        'contrast_effect_pretrained_minus_none/' + left + '_minus_' + right for left, right in PAIRS}
    require(set(primary['between_arms']) == expected_between and len(expected_between) == 27, 'Between27 coverage differs')
    for name in sorted(expected_between):
        recomputed['between_arms'][name] = {}
        for key in QUALITY:
            category, rest = name.split('/', 1)
            if category == 'absolute_pretrained_minus_none':
                effects = matrix(p, rest, key) - matrix(n, rest, key)
            else:
                left, right = next((a, b) for a, b in PAIRS if a + '_minus_' + b == rest)
                effects = (matrix(p, left, key) - matrix(p, right, key)) - (matrix(n, left, key) - matrix(n, right, key))
            r = paired(effects, indices, counts, key)
            recomputed['between_arms'][name][key] = r
            check_pair(r, primary['between_arms'][name][key], f'between/{name}/{key}', names, scenes, checks, 'between_ci_endpoints')
    require(checks.categories['raw_reported_means']['scalar_checks'] == 372, 'Mean coverage is not 2x31x6')
    require(checks.categories['within_ci_endpoints']['scalar_checks'] == 192 and
            checks.categories['between_ci_endpoints']['scalar_checks'] == 216, 'CI coverage is not 408 endpoints')
    replay = {}
    for arm, filename in [('pretrained', args.legacy_pretrained), ('none', args.legacy_none)]:
        if filename:
            paths['legacy_' + arm] = filename
            old = load(filename)
            new_origin = original[arm]['prepared_manifest']['source_cache_provenance']
            old_origin = old['base_cache_provenance']
            for field in ('base_checkpoint_sha256', 'base_state_sha256', 'raft_weights_sha256', 'raft_state_sha256'):
                require(new_origin[field] == old_origin[field], f'Replay lineage differs: {arm}/{field}')
            replay[arm] = drift(arms[arm], old, arm)
    temp_gt = {'available': False, 'tolerance': args.gt_tolerance}
    if args.tempretinex:
        paths['tempretinex'] = args.tempretinex
        temp = load(args.tempretinex)
        tn, _, tr = rows_by_name(temp)
        require(tn == names and temp['domain'] == 'bvi' and temp['split'] == 'val' and temp['n'] == 16, 'Temp target differs')
        tg = np.asarray([tr[x]['gt_tlp'] for x in names])
        delta = tg - p['values']['base']['gt_tlp']
        temp_gt = {'available': True, 'tolerance': args.gt_tolerance,
                   'max_abs_difference': float(abs(delta).max()), 'mean_abs_difference': float(abs(delta).mean()),
                   'passed': bool(abs(delta).max() <= args.gt_tolerance),
                   'per_sequence_difference': dict(zip(names, delta.tolist()))}
        checks.compare('tempretinex_gt', 'Temp/P GT tLP', tg, p['values']['base']['gt_tlp'], args.gt_tolerance)
    return {'purpose': 'Independent raw-JSON NumPy mathematical verification; no imports of project analysis code',
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'script_sha256': sha(__file__), 'input_files': {k: {'path': str(Path(v).resolve()), 'sha256': sha(v)} for k, v in paths.items()},
            'numpy_version': np.__version__, 'n_sequences': 16, 'n_scenes': 8, 'head_seeds': [0, 1, 2],
            'bootstrap': {'rng_seed': 0, 'resamples': 10000, 'unit': 'whole scene with both conditions',
                          'head_seeds_averaged_first': True, 'shared_draws': True,
                          'implementation': 'scene multiplicity matrix @ seed-averaged scene means /8; percentile2.5/97.5',
                          'draw_matrix_sha256': hashlib.sha256(draws.tobytes()).hexdigest()},
            'checks': checks.categories, 'numeric_failures': checks.failures, 'passed': not checks.failures,
            'computed': recomputed, 'legacy_replay_drift': replay, 'tempretinex_gt_reference': temp_gt,
            'scope_limitations': ['Mathematical verification of supplied completed raw files, not causal-mechanism or SOTA evidence',
                                  'Intervals conditional on three fitted added-head models and one historical base per arm',
                                  'No causal pretrained-prior isolation across historically different base initializations',
                                  'Replay drift is measured rather than required to be zero; Temp GT tolerance is explicit']}


def self_test():
    names = [f'BVI/scene{i:02d}__{c}fps' for i in range(8) for c in (10, 20)]
    indices = scene_indices(names, sorted({scene(n) for n in names}))
    draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
    counts = np.stack([np.bincount(row, minlength=8) for row in draws])
    constant = paired(np.full((3, 16), .25), indices, counts, 'psnr_y')
    assert constant['scene_ci95'] == [.25, .25] and constant['difference'] == .25
    effects = np.arange(48, dtype=float).reshape(3, 16) / 17
    actual = paired(effects, indices, counts, 'lpips')
    scenes = np.asarray([effects.mean(0)[idx].mean() for idx in indices])
    independent_gather = np.quantile(scenes[draws].mean(1), [.025, .975])
    np.testing.assert_allclose(actual['scene_ci95'], independent_gather, atol=1e-14, rtol=0)
    assert actual['selector_seeds_better'] == 0
    # Opposite seed effects cancel before resampling; they are not 24 independent scenes.
    cancel = np.stack([np.ones(16), -np.ones(16), np.zeros(16)])
    assert paired(cancel, indices, counts, 'psnr_y')['scene_ci95'] == [0, 0]
    assert len(METHODS) == 31 and len(LABELS) == 15 and len(PAIRS) == 12
    print('PASS synthetic algebra/scene-pairing/seed-averaging/coverage self-test; no experiment results')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--none', type=Path)
    parser.add_argument('--main-analysis', type=Path)
    parser.add_argument('--legacy-pretrained', type=Path)
    parser.add_argument('--legacy-none', type=Path)
    parser.add_argument('--tempretinex', type=Path)
    parser.add_argument('--gt-tolerance', type=float, default=1e-6)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(all(getattr(args, k) for k in ('pretrained', 'none', 'main_analysis', 'out')), 'Need actual P/N raw, main analysis, and output paths')
    require(args.gt_tolerance >= 0 and np.isfinite(args.gt_tolerance), 'Invalid GT tolerance')
    require(not args.out.exists(), 'Use a new output path')
    result = verify(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'passed': result['passed'], 'out': str(args.out), 'sha256': sha(args.out),
                      'raw_mean_checks': result['checks']['raw_reported_means']['scalar_checks'],
                      'ci_endpoints': result['checks']['within_ci_endpoints']['scalar_checks'] + result['checks']['between_ci_endpoints']['scalar_checks'],
                      'max_numeric_error': max(x['max_abs_error'] for x in result['checks'].values())}))
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
