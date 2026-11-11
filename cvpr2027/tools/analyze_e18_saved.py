"""Paired E18 saved-reconstruction audit; never selects checkpoints or settings."""
import hashlib
import json
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AUDIT = REPO / 'results' / 'protocol_audit_20261111'
NEW = AUDIT / 'reevaluated'
OLD = REPO / 'results' / 'cvpr_converged'
METRICS = ['psnr_y', 'lpips', 'psnr_y_gtmean', 'd_tlp', 'd_mabd', 'd_flicker', 'd_warp_err']


def read(path):
    result = json.loads(path.read_text())
    result['_source_path'] = str(path.relative_to(REPO.parent))
    result['_source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['split'] == 'test' and result['n_sequences'] == 39
    assert len(result['per_sequence']) == 39
    assert len({row['name'] for row in result['per_sequence']}) == 39
    return result


def score(row, metric):
    if metric.startswith('d_'):
        key = metric[2:]
        if row.get(key) is None or row.get('gt_' + key) is None:
            return None
        return abs(row[key] - row['gt_' + key])
    return row.get(metric)


def paired(runs, baseline, description):
    assert len(runs) == len(baseline)
    for a, b in zip(runs, baseline):
        assert a['seed'] == b['seed']
    names = sorted(row['name'] for row in runs[0]['per_sequence'])
    assert all(sorted(row['name'] for row in r['per_sequence']) == names for r in runs + baseline)
    ar = [{row['name']: row for row in r['per_sequence']} for r in runs]
    br = [{row['name']: row for row in r['per_sequence']} for r in baseline]
    clusters = {}
    for i, name in enumerate(names):
        clusters.setdefault(re.sub(r'_2$', '', name), []).append(i)
    groups = list(clusters.values())
    # Shared draws across metrics. The sequence and scene streams both begin at seed 0.
    sequence_draws = np.random.default_rng(0).integers(0, len(names), size=(10000, len(names)))
    cluster_choices = np.random.default_rng(0).integers(0, len(groups), size=(10000, len(groups)))
    cluster_draws = [np.concatenate([groups[i] for i in choice]) for choice in cluster_choices]
    result = {
        'description': description,
        'difference_direction': 'corrected E18 minus matched prior',
        'seeds': [r['seed'] for r in runs],
        'n_sequences': len(names),
        'n_scene_clusters': len(groups),
        'sources': [{k: r[k] for k in ('_source_path', '_source_sha256', 'seed', 'aggregate')} for r in runs],
        'baseline_sources': [{k: r[k] for k in ('_source_path', '_source_sha256', 'seed', 'aggregate')} for r in baseline],
        'metrics': {},
    }
    for metric in METRICS:
        av = [[score(r[n], metric) for n in names] for r in ar]
        bv = [[score(r[n], metric) for n in names] for r in br]
        if any(v is None for row in av + bv for v in row):
            result['metrics'][metric] = {'available': False}
            continue
        av, bv = np.asarray(av), np.asarray(bv)
        assert np.isfinite(av).all() and np.isfinite(bv).all(), metric
        diff = av - bv
        if metric.startswith('d_'):
            key = metric[2:]
            raw = lambda rows, k: np.asarray([[r[n][k] for n in names] for r in rows], dtype=float)
            a_collapsed = np.abs(raw(ar, key).mean(axis=0) - raw(ar, 'gt_' + key).mean(axis=0))
            b_collapsed = np.abs(raw(br, key).mean(axis=0) - raw(br, 'gt_' + key).mean(axis=0))
        else:
            a_collapsed, b_collapsed = av.mean(axis=0), bv.mean(axis=0)
        avg_diff = a_collapsed - b_collapsed
        seq_ci = np.percentile(avg_diff[sequence_draws].mean(axis=1), [2.5, 97.5])
        cluster_ci = np.percentile([avg_diff[idx].mean() for idx in cluster_draws], [2.5, 97.5])
        per_seed = diff.mean(axis=1)
        consistent = bool(np.all(per_seed > 0) or np.all(per_seed < 0))
        result['metrics'][metric] = {
            'available': True,
            'corrected_mean': float(a_collapsed.mean()),
            'baseline_mean': float(b_collapsed.mean()),
            'mean_difference': float(avg_diff.mean()),
            'per_seed_differences': per_seed.tolist(),
            'sequence_ci95': seq_ci.tolist(),
            'scene_cluster_ci95': cluster_ci.tolist(),
            'same_sign_across_seeds': consistent if len(runs) > 1 else None,
            'ci_excludes_zero': bool(seq_ci[0] > 0 or seq_ci[1] < 0),
            'interpretation_limit': 'Two observed training seeds only' if len(runs) > 1 else 'One seed and training-hardware difference; conditional checkpoint comparison only',
        }
        if metric.startswith('d_'):
            result['metrics'][metric]['per_seed_abs_gap_then_average_diagnostic'] = {
                'definition': 'Alternative diagnostic only: take each seed absolute gap before averaging seeds',
                'corrected_mean': float(av.mean()), 'baseline_mean': float(bv.mean()),
                'mean_difference': float(diff.mean()),
            }
    return result


hybrid = [read(NEW / f'e18_hybrid_s{s}_saved.json') for s in (0, 1)]
all_layers = read(NEW / 'e18_layers_all_s2_saved.json')
prior = [read(OLD / 'indomain' / f'e17_small_pretrained_s{s}.json') for s in (0, 1)]
prior2 = read(OLD / 'indomain' / 'e19_small_pretrained_s2.json')
assert [r['seed'] for r in hybrid] == [0, 1] and all_layers['seed'] == 2
assert all(r['reconstruction']['mode'] == 'saved' and r['reconstruction']['kv'] == 'hybrid' for r in hybrid)
assert all_layers['reconstruction']['mode'] == 'saved' and all_layers['reconstruction']['layers'] == 'all'

out = {
    'analysis': 'Inference-only recovery of available final-step E18 checkpoints; no tuning or checkpoint selection',
    'bootstrap': {'replicates': 10000, 'rng_seed': 0, 'shared_draws_across_metrics': True, 'sequence_and_scene_rng_streams': 'independently initialized with seed 0', 'scene_rule': 'strip terminal _2; resample clusters retaining every member; sequence-weighted statistic'},
    'temporal_score': 'Primary matches repaired aggregate.py: average raw prediction and GT across seeds within each arm and sequence, then absolute gap and paired difference. Per-seed signs use within-seed absolute gaps. Alternative mean-of-per-seed-absolute-gaps is separately labeled diagnostic.',
    'hybrid_vs_matched_prior': paired(hybrid, prior, 'Hybrid kv, seeds 0 and 1 versus corresponding full-token per-tubelet prior'),
    'all_layers_vs_prior_seed2': paired([all_layers], [prior2], 'All-layer checkpoint directory ends npu_s0 but saved seed is 2; prior seed2 was trained on different hardware'),
    'legacy_aggregate_for_audit_only': {name: read(OLD / 'injection' / f'{name}.json')['aggregate'] for name in ('e18_hybrid_s0', 'e18_hybrid_s1', 'e18_layers_all_npu_s0')},
}
(AUDIT / 'e18_comparisons.json').write_text(json.dumps(out, indent=2) + '\n')
print(f"Wrote {AUDIT / 'e18_comparisons.json'}")
