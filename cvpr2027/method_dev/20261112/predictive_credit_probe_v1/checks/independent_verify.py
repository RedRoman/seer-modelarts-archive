"""Independent scalar-only replay; intentionally imports no producer/analysis code."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('subset', 'critic', 'values', 'analysis', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    checks = 0

    def check(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1

    subset = json.loads(args.subset.read_text())
    critic = json.loads(args.critic.read_text())
    result = json.loads(args.analysis.read_text())
    with np.load(args.values, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    check(len(subset['entries']) == critic['count'] == len(result['per_scene']) == 12, 'twelve fixed scenes')
    check(critic['gt_storage_entries_read'] == 0, 'GT-free critic')
    check(not critic['quality_scores_computed'] and not critic['raw_videos_saved'], 'scalar critic only')
    check(len(values['clip_index']) == 3072, 'complete patch coverage')
    for key, vector in values.items():
        check(vector.shape == (3072,) and np.isfinite(vector).all(), 'finite scalar column ' + key)
    check(np.array_equal(values['mse_base'], values['mse_sameframe_zero']), 'zero-action exact identity')
    formulas = {'u1': ('mse_base', 'mse_single1'), 'u2': ('mse_base', 'mse_single2'),
                'u12': ('mse_base', 'mse_pair'), 'u1_given2': ('mse_single2', 'mse_pair'),
                'u2_given1': ('mse_single1', 'mse_pair')}
    for name, (left, right) in formulas.items():
        check(np.array_equal(values[name], values[left] - values[right]), 'utility subtraction ' + name)
    clips = {(row['scene'], row['name']): row for row in critic['clips']}
    reported = {row['scene']: row for row in result['per_scene']}
    auc_rows = []
    for i, entry in enumerate(subset['entries']):
        check(entry['scene'] in subset['split']['fit'] and entry['scene'] not in subset['split']['internal_holdout'], 'FIT guard')
        item = clips[(entry['scene'], entry['name'])]
        path = args.critic.parent / item['path']
        check(sha(path) == item['sha256'], 'critic clip hash')
        data = json.loads(path.read_text())
        check(data['rgb_lq_tensor_sha256'] == entry['rgb_lq_tensor_sha256'], 'canonical RGB identity')
        check(data['target_frames'] == [8, 9] and data['donor_frames'] == [[6, 7], [10, 11]], 'fixed time mapping')
        check(data['patch_yx'] == [[y, x] for y in range(16) for x in range(16)], 'complete patch order')
        chosen = values['clip_index'] == i
        check(int(chosen.sum()) == 256, 'one scene one clip')
        check(np.array_equal(values['patch_y'][chosen], np.repeat(np.arange(16), 16)), 'value y order')
        check(np.array_equal(values['patch_x'][chosen], np.tile(np.arange(16), 16)), 'value x order')
        support = (values['feature_valid1'][chosen] >= .5) & (values['feature_valid2'][chosen] >= .5)
        utilities = np.stack([values['u1_given2'][chosen], values['u2_given1'][chosen]], axis=1)
        eligible = np.repeat(support[:, None], 2, axis=1) & (np.abs(utilities) > 1e-10)
        labels = utilities[eligible] > 0
        scores = {}
        for mode, name in [('trained', 'trained_conditional'), ('random', 'random_conditional'),
                           ('wrong_time', 'wrong_time_conditional')]:
            losses = np.asarray(data['losses'][mode], dtype=float)
            check(losses.shape == (4, 256) and np.isfinite(losses).all() and losses.min() >= 0 and losses.max() <= 4.0001, 'unit loss geometry ' + mode)
            scores[name] = np.stack([losses[2] - losses[3], losses[1] - losses[3]], axis=1)
        scores['ema_cosine'] = np.asarray(data['static_full_ema_flow_aligned_cosine']).T
        for feature, name in [('lq', 'negative_lq_mse'), ('hvi', 'negative_hvi_mse')]:
            scores[name] = np.stack([values[f'feature_{feature}{d}'][chosen] for d in (1, 2)], axis=1)
        row = reported[entry['scene']]
        check(row['eligible_patches'] == int(support.sum()), 'eligible patch count')
        check(row['positive_donor_labels'] == int(labels.sum()), 'positive label count')
        check(row['negative_donor_labels'] == int((~labels).sum()), 'negative label count')
        check(row['excluded_zero_utility_donors'] == int((np.repeat(support[:, None], 2, axis=1) & ~eligible).sum()), 'utility tie count')
        auc = {}
        for name, vector in scores.items():
            score = vector[eligible]
            positive, negative = score[labels], score[~labels]
            # Pairwise wins/ties is independent of the analysis' rank-sum method.
            score_auc = None if not len(positive) or not len(negative) else float(
                ((positive[:, None] > negative[None, :]).sum() + .5 * (positive[:, None] == negative[None, :]).sum())
                / (len(positive) * len(negative)))
            actual = row['auc'][name]
            check((actual is None and score_auc is None) or
                  (actual is not None and score_auc is not None and abs(actual - score_auc) < 1e-12), 'pairwise AUC ' + name)
            auc[name] = score_auc
        auc_rows.append(auc)
        local_loss = np.asarray(data['target_local_vs_full_normalized_l2'])
        local_cos = np.asarray(data['target_local_vs_full_cosine'])
        check(np.max(np.abs(local_loss - (2 - 2 * local_cos))) < 2e-5, 'reference unit-vector geometry')
    for name in auc_rows[0]:
        defined = [row[name] for row in auc_rows if row[name] is not None]
        reported_macro = result['primary_macro_auc'][name]
        check(reported_macro['defined_scenes'] == len(defined), 'macro scene count ' + name)
        check((not defined and reported_macro['mean'] is None) or
              abs(reported_macro['mean'] - np.mean(defined)) < 1e-12, 'equal-scene macro ' + name)
    # Common scene definition and paired means are replayed without invoking its bootstrap.
    common = [row for row in auc_rows if all(value is not None for value in row.values())]
    for name, comparison in result['primary_gate']['comparisons'].items():
        check(len(comparison['matched_scene_names']) == len(common), 'common-scene comparison')
        if common:
            delta = np.mean([row['trained_conditional'] - row[name] for row in common])
            check(abs(comparison['bootstrap']['mean'] - delta) < 1e-12, 'paired scene difference ' + name)
            check(comparison['bootstrap']['resampling_unit'] == 'complete_scene', 'scene bootstrap ' + name)
            check(comparison['bootstrap']['repeats'] == 20000 and comparison['bootstrap']['seed'] == 20261123, 'frozen bootstrap ' + name)
    output = {'status': 'passed', 'checks': checks, 'AUC_method': 'direct positive-negative pairwise wins and half ties',
              'analysis_imported': False, 'torch_imported': False, 'GT_or_tensor_archives_opened': 0,
              'files': {name: sha(getattr(args, name)) for name in ('subset', 'critic', 'values', 'analysis')}}
    with args.output.open('x') as handle:
        json.dump(output, handle, indent=2)
        handle.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
