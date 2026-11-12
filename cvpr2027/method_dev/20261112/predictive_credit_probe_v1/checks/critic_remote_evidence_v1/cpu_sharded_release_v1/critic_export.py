"""Frozen LQ-only, genuinely masked V-JEPA2 coalition critic for a source probe.

The losses measure prediction of observed noisy LQ, not clean RGB risk. Native
256 crops retain their geometry; only the official RGB mean/std is reused.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ASSET = Path('/cache/zihaohe/seer_matched_predictor_assets_20261112_v1')
COLOR_RELEASE = Path('/cache/zihaohe/seer_color_contract_20261112/export_release_v1')
SOURCE = Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0')
ASSET_SHA = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'
READER_SHA = '26c0c36656ad7b7d3b0ce04275578a45286c27931a476c9dcfb1ce1981e69db2'
COLOR_SHA = '7815887150b2e92ed7c2cf4c30f334a4a0c435d58b4ba017fd1b746a97dcab60'
BGR_SHA = 'fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70'
COALITIONS = ('empty', 'donor1', 'donor2', 'both')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mask_plan(tubelets=8, target=4, donors=(3, 5), patches=256):
    if len(set((target, *donors))) != 3 or len(donors) != 2:
        raise ValueError('Target and physical donors must be distinct; duplicate donors require deduplication')
    if any(value < 0 or value >= tubelets for value in (target, *donors)):
        raise ValueError('Out-of-range tubelet')
    common = [value for value in range(tubelets) if value not in (target, *donors)]
    selected = (common, common+[donors[0]], common+[donors[1]], common+list(donors))
    def ids(values):
        return [value*patches+patch for value in sorted(values) for patch in range(patches)]
    return {'target_ids': ids([target]), 'donor_ids': [ids([value]) for value in donors],
            'common_tubelets': common, 'context_ids': dict(zip(COALITIONS, map(ids, selected)))}


def normalize_rgb(low):
    import torch
    if low.ndim != 5 or low.shape[1] != 3 or low.dtype != torch.float32:
        raise ValueError('Expected float32 B,3,T,H,W RGB')
    if not bool(torch.isfinite(low).all()) or float(low.min()) < 0 or float(low.max()) > 1:
        raise ValueError('Expected finite unit-range LQ')
    mean = low.new_tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = low.new_tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
    return (low-mean)/std


def normalized_loss(prediction, target):
    import torch.nn.functional as F
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('Prediction/target contract mismatch')
    return ((F.normalize(prediction, dim=-1, eps=1e-12)-
             F.normalize(target, dim=-1, eps=1e-12))**2).sum(-1)


def cosine(first, second):
    import torch.nn.functional as F
    return F.cosine_similarity(first, second, dim=-1, eps=1e-12)


def wrong_time_content(video):
    """Bijection over all nontarget tubelets, preserving H and within-tubelet order."""
    if video.shape[2] != 16:
        raise ValueError('Wrong-time control requires the declared16 frames')
    selected = [0, 1, 2, 3, 5, 6, 7]
    result = video.clone()
    for index, destination in enumerate(selected):
        source = selected[(index+3) % len(selected)]
        result[:, :, 2*destination:2*destination+2] = video[:, :, 2*source:2*source+2]
    return result


def aligned_cosine(target, donor, source_xy):
    """Sample a16x16 donor descriptor grid at mean valid flow source coordinates.

    source_xy uses zero-based source-pixel centers. A coarse-token center p maps
    to raw-pixel center16*p+7.5. No descriptor is scored against clean GT.
    """
    import torch
    import torch.nn.functional as F
    xy = torch.as_tensor(source_xy, dtype=target.dtype, device=target.device)
    if list(xy.shape) != [256, 2] or not bool(torch.isfinite(xy).all()):
        raise ValueError('Expected256 finite source-pixel xy centers')
    grid = (2*(xy+.5)/256-1).view(1, 16, 16, 2)
    donor_grid = donor.reshape(1, 16, 16, -1).permute(0, 3, 1, 2)
    aligned = F.grid_sample(donor_grid, grid, mode='bilinear', padding_mode='border', align_corners=False)
    aligned = aligned.permute(0, 2, 3, 1).reshape(1, 256, -1)
    return cosine(target, aligned)


def predictor_constructor(predictor_module):
    return predictor_module.vit_predictor(img_size=(256, 256), patch_size=16, num_frames=64,
        tubelet_size=2, embed_dim=1024, predictor_embed_dim=384, out_embed_dim=None,
        depth=12, num_heads=12, num_mask_tokens=10, use_mask_tokens=True,
        use_rope=True, uniform_power=False, use_sdpa=True, use_silu=False, wide_silu=True)


def load_models():
    import torch
    helper = load_file('frozen_matched_asset_helper', ASSET/'inspect_and_smoke.py')
    receipt = json.loads((ASSET/'evidence/download_v3_receipt.json').read_text())
    checkpoint = ASSET/'weights/vitl.pt'
    assert receipt['status'] == 'authenticated' and receipt['sha256'] == ASSET_SHA
    assert checkpoint.stat().st_size == 5127726842 and digest(checkpoint) == ASSET_SHA
    upstream = json.loads((ASSET/'evidence/upstream_manifest.json').read_text())
    for name, expected in upstream['files'].items():
        assert digest(ASSET/'upstream'/name) == expected, name
    sys.path.insert(0, str(ASSET/'upstream'))
    from src.models import vision_transformer, predictor as predictor_module
    assert Path(vision_transformer.__file__).resolve() == (ASSET/'upstream/src/models/vision_transformer.py').resolve()
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    encoder = vision_transformer.vit_large(img_size=(256, 256), patch_size=16, num_frames=64,
        tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False, use_rope=True)
    predictor = predictor_constructor(predictor_module)
    omitted = {'encoder': helper.load_strict_with_known_static_position(
        encoder, helper.clean(payload['target_encoder']), 'pos_embed'),
        'predictor': helper.load_strict_with_known_static_position(
        predictor, helper.clean(payload['predictor']), 'predictor_pos_embed')}
    with torch.random.fork_rng():
        torch.manual_seed(20261122)
        random_predictor = predictor_constructor(predictor_module)
    assert list(random_predictor.state_dict()) == list(predictor.state_dict())
    for model in (encoder, predictor, random_predictor):
        model.float().eval().requires_grad_(False)
    states = {name: helper.state_sha(model.state_dict()) for name, model in
              [('encoder', encoder), ('trained', predictor), ('random', random_predictor)]}
    return encoder, predictor, random_predictor, helper, states, omitted


def ids_tensor(ids, device):
    import torch
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def evaluate_clip(video, encoder, trained, random, plan, observable):
    """All encoder masking occurs before block 0, through the frozen official API."""
    import torch
    device = video.device
    target_ids = ids_tensor(plan['target_ids'], device)
    block_shapes = []
    handle = encoder.blocks[0].register_forward_pre_hook(
        lambda module, args: block_shapes.append(list(args[0].shape)))
    losses = {'trained': [], 'random': [], 'wrong_time': []}
    with torch.inference_mode():
        target = encoder(video, masks=[target_ids])
        full = encoder(video)
        full_target = full.index_select(1, target_ids[0])
        full_cos, local_cos, aligned_full_cos = [], [], []
        for donor_index, donor_ids in enumerate(plan['donor_ids']):
            donor_ids = ids_tensor(donor_ids, device)
            donor_full = full.index_select(1, donor_ids[0])
            full_cos.append(cosine(full_target, donor_full))
            aligned_full_cos.append(aligned_cosine(full_target, donor_full, observable['patch_grid_xy'][donor_index]))
            local_cos.append(cosine(target, encoder(video, masks=[donor_ids])))
        for label in COALITIONS:
            context_ids = ids_tensor(plan['context_ids'][label], device)
            context = encoder(video, masks=[context_ids])
            for key, predictor in [('trained', trained), ('random', random)]:
                prediction = predictor(context, masks_x=[context_ids], masks_y=[target_ids], mask_index=0)
                loss = normalized_loss(prediction, target)
                assert bool(torch.isfinite(loss).all()) and float(loss.min()) >= 0 and float(loss.max()) <= 4.00001
                losses[key].append(loss[0].cpu().tolist())
        wrong_video = wrong_time_content(video)
        for label in COALITIONS:
            context_ids = ids_tensor(plan['context_ids'][label], device)
            context = encoder(wrong_video, masks=[context_ids])
            prediction = trained(context, masks_x=[context_ids], masks_y=[target_ids], mask_index=0)
            loss = normalized_loss(prediction, target)
            assert bool(torch.isfinite(loss).all()) and float(loss.min()) >= 0 and float(loss.max()) <= 4.00001
            losses['wrong_time'].append(loss[0].cpu().tolist())
        result = {'losses': losses,
                  'static_full_ema_cosine': [value[0].cpu().tolist() for value in full_cos],
                  'static_full_ema_flow_aligned_cosine': [value[0].cpu().tolist() for value in aligned_full_cos],
                  'static_local_ema_cosine': [value[0].cpu().tolist() for value in local_cos],
                  'target_local_vs_full_cosine': cosine(target, full_target)[0].cpu().tolist(),
                  'target_local_vs_full_normalized_l2': normalized_loss(target, full_target)[0].cpu().tolist()}
    handle.remove()
    expected = [len(plan['target_ids']), video.shape[2]//2*256,
                *map(len, plan['donor_ids']), *[len(plan['context_ids'][key]) for key in COALITIONS]*2]
    assert [shape[1] for shape in block_shapes] == expected, block_shapes
    result['pre_attention_first_block_shapes'] = block_shapes
    return result


def smoke(encoder, trained, random, output):
    import torch
    # Four synthetic frames, retained absolute target IDs 256..271.
    video = torch.linspace(0, 1, 3*4*256*256, dtype=torch.float32).reshape(1, 3, 4, 256, 256)
    video = normalize_rgb(video)
    context_ids = torch.arange(32, dtype=torch.long).unsqueeze(0)
    target_ids = torch.arange(256, 272, dtype=torch.long).unsqueeze(0)
    def once(device):
        models = [model.to(device) for model in (encoder, trained, random)]
        clip, context, target = video.to(device), context_ids.to(device), target_ids.to(device)
        shapes = []
        handle = encoder.blocks[0].register_forward_pre_hook(lambda module, args: shapes.append(list(args[0].shape)))
        with torch.inference_mode():
            x = encoder(clip, masks=[context])
            y = encoder(clip, masks=[target])
            changed = clip.clone()
            changed[:, :, 2:] += .12345
            xx = encoder(changed, masks=[context])
            predictions = [model(x, masks_x=[context], masks_y=[target], mask_index=0) for model in models[1:]]
            altered = trained(xx, masks_x=[context], masks_y=[target], mask_index=0)
            assert torch.equal(x, xx) and torch.equal(predictions[0], altered)
            losses = torch.stack([normalized_loss(prediction, y) for prediction in predictions]).cpu()
        handle.remove()
        assert shapes == [[1, 32, 1024], [1, 16, 1024], [1, 32, 1024]], shapes
        assert bool(torch.isfinite(losses).all())
        return losses, {'block_shapes': shapes, 'masked_context_exact_invariance': True,
                        'masked_prediction_exact_invariance': True, 'target_token_ids': target_ids.tolist()}
    cpu, cpu_record = once('cpu')
    import torch_npu  # noqa: F401 -- explicit NPU registration after CPU reference
    torch.npu.set_device('npu:0')
    npu, npu_record = once('npu:0')
    difference = (cpu-npu).abs()
    record = {'status': 'passed', 'dataset_pixels_read': 0, 'gt_pixels_read': 0,
              'cpu': cpu_record, 'npu': npu_record, 'cpu_losses': cpu.tolist(), 'npu_losses': npu.tolist(),
              'normalized_loss_abs_max': float(difference.max()), 'normalized_loss_abs_mean': float(difference.mean()),
              'predeclared_max_tolerance': .01, 'predeclared_mean_tolerance': .001}
    assert record['normalized_loss_abs_max'] <= .01 and record['normalized_loss_abs_mean'] <= .001, record
    write_new(output, record)
    return record


def check_release(release):
    frozen = json.loads((release/'critic_release_manifest.json').read_text())
    for path, expected in frozen['files'].items():
        assert digest(release/path) == expected, path
    for path, expected in frozen['absolute_dependencies'].items():
        assert digest(Path(path)) == expected, path
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'export'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    release = args.release.resolve()
    frozen = check_release(release)
    import torch
    torch.set_num_threads(4)
    encoder, trained, random, helper, before, omitted = load_models()
    if args.mode == 'smoke':
        smoke(encoder, trained, random, args.output)
    else:
        passed = json.loads((release.parent/'checks/critic_synthetic_smoke.json').read_text())
        assert passed['status'] == 'passed'
        subsets = json.loads((release/'subsets.json').read_text())
        selected = subsets['entries']
        assert len(selected) == 12 and len({item['scene'] for item in selected}) == 12
        assert all(item['name'] == item['scene'] for item in selected), 'Only pass1 names permitted'
        bgr = json.loads((COLOR_RELEASE/'proofs/source_color_receipt_closed.json').read_text())
        assert bgr['status'] == 'closed' and bgr['native_source_color'] == 'BGR'
        reader = load_file('frozen_selective_reader', COLOR_RELEASE/'dependencies/cache_hvi_source.py')
        color = load_file('frozen_explicit_color', COLOR_RELEASE/'dependencies/color_contract.py')
        if args.output.exists():
            raise FileExistsError('Refusing to reuse critic output directory')
        args.output.mkdir(parents=True)
        import torch_npu  # noqa: F401
        torch.npu.set_device('npu:0')
        encoder.to('npu:0'); trained.to('npu:0'); random.to('npu:0')
        plan = mask_plan()
        observable_path = release/'observable_sidecar.json'
        observable_payload = json.loads(observable_path.read_text())
        assert observable_payload['gt_storage_entries_read'] == 0
        observable_by_name = {row['name']: row for row in observable_payload['clips']}
        manifest = []
        for row in selected:
            began = time.monotonic()
            path = SOURCE/row['source_file']
            assert digest(path) == row['source_sha256']
            name, scene, low, lineage = reader.read_lq_only(path)
            assert (name, scene) == (row['name'], row['scene'])
            rgb = color.convert_video(low, source_order='BGR', target_order='RGB')
            rgb_sha = reader.tensor_digest(rgb)
            assert lineage['lq_tensor_sha256'] == row['native_lq_tensor_sha256']
            assert rgb_sha == row['rgb_lq_tensor_sha256']
            result = evaluate_clip(normalize_rgb(rgb).to('npu:0'), encoder, trained, random, plan, observable_by_name[name])
            result.update({'name': name, 'scene': scene, 'cache_path': row['source_file'],
                           'source_archive_sha256': row['source_sha256'], 'rgb_lq_tensor_sha256': rgb_sha,
                           'observable_sidecar_sha256': digest(observable_path),
                           'lineage': lineage, 'mask_plan': plan, 'coalition_order': list(COALITIONS),
                           'patch_yx': [[i//16, i%16] for i in range(256)],
                           'target_frames': [8, 9], 'donor_frames': [[6, 7], [10, 11]],
                           'elapsed_seconds': time.monotonic()-began})
            destination = args.output/(Path(row['source_file']).stem+'.json')
            write_new(destination, result)
            manifest.append({'path': destination.name, 'sha256': digest(destination), 'name': name, 'scene': scene,
                             'rgb_lq_tensor_sha256': rgb_sha, 'elapsed_seconds': result['elapsed_seconds']})
            print(json.dumps(manifest[-1]), flush=True)
        write_new(args.output/'manifest.json', {'schema_version': 1, 'clips': manifest,
                  'count': 12, 'gt_storage_entries_read': 0, 'quality_scores_computed': False,
                  'raw_videos_saved': False, 'release_sha256': digest(release/'critic_release_manifest.json')})
    after = {name: helper.state_sha(model.state_dict()) for name, model in
             [('encoder', encoder), ('trained', trained), ('random', random)]}
    assert before == after
    check_release(release)
    assert digest(ASSET/'weights/vitl.pt') == ASSET_SHA
    write_new(release.parent/'checks'/('critic_'+args.mode+'_complete.json'), {
        'status': 'passed', 'mode': args.mode, 'state_before': before, 'state_after': after,
        'states_unchanged': True, 'asset_sha256': ASSET_SHA, 'known_static_position_omissions': omitted,
        'dependencies_unchanged': True, 'gt_storage_entries_read': 0,
        'quality_scores_computed': False, 'elapsed_seconds': time.monotonic()-start,
        'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
