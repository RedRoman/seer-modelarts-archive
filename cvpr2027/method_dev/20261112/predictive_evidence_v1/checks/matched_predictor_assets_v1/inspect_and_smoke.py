"""Strict CPU synthetic V-JEPA2-L interface check, using a frozen local source tree."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/cache/zihaohe/seer_matched_predictor_assets_20261112_v1')
SHA = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'
SIZE = 5127726842


def file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def clean(state):
    result = {}
    for key, value in state.items():
        canonical = key.replace('module.', '').replace('backbone.', '')
        if canonical in result:
            raise ValueError('Checkpoint key-cleaning collision: '+canonical)
        result[canonical] = value
    return result


def state_sha(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(key.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def load_strict_with_known_static_position(module, state, position_key):
    """Reject arbitrary mismatch; only an unused frozen absolute position grid may be omitted."""
    selected = dict(state)
    omitted = {}
    if position_key in selected and position_key not in module.state_dict():
        position = selected.pop(position_key)
        omitted[position_key] = {'shape': list(position.shape), 'dtype': str(position.dtype),
            'sha256': hashlib.sha256(position.cpu().contiguous().numpy().tobytes()).hexdigest(),
            'reason': 'Official RoPE constructor has no absolute-position parameter; this static checkpoint grid is unused'}
    result = module.load_state_dict(selected, strict=True, assign=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    return omitted


def main():
    began = time.monotonic()
    receipt = json.loads((ROOT/'evidence/download_v3_receipt.json').read_text())
    primary = json.loads((ROOT/'checksum_primary_receipt.json').read_text())
    checkpoint = ROOT/'weights/vitl.pt'
    assert receipt['status'] == 'authenticated' and receipt['sha256'] == primary['sha256'] == SHA
    assert receipt['bytes'] == primary['bytes'] == SIZE
    assert checkpoint.stat().st_size == SIZE and file_sha(checkpoint) == SHA
    frozen = json.loads((ROOT/'evidence/upstream_manifest.json').read_text())
    upstream = ROOT/'upstream'
    for name, expected in frozen['files'].items():
        assert file_sha(upstream/name) == expected, name
    # No torch.hub invocation and no network model loader. Only already frozen local modules.
    sys.path.insert(0, str(upstream))
    import torch
    torch.set_num_threads(4)
    from src.models import vision_transformer, predictor as predictor_module
    assert Path(vision_transformer.__file__).resolve() == (upstream/'src/models/vision_transformer.py').resolve()
    assert Path(predictor_module.__file__).resolve() == (upstream/'src/models/predictor.py').resolve()
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    assert isinstance(payload, dict) and 'target_encoder' in payload and 'predictor' in payload
    encoder_state, predictor_state = clean(payload['target_encoder']), clean(payload['predictor'])
    tensors = {'target_encoder': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                                for key, value in encoder_state.items()},
               'predictor': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                             for key, value in predictor_state.items()}}
    online = clean(payload['encoder']) if 'encoder' in payload else None
    if online is not None:
        tensors['encoder_online_inventory_only'] = {
            key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in online.items()}
    write_new(ROOT/'evidence/checkpoint_tensor_manifest.json', {
        'checkpoint_sha256': SHA, 'top_level_keys': list(payload), 'selected_encoder_key': 'target_encoder',
        'clean_key_rule': "replace('module.','').replace('backbone.','') per frozen official hub code",
        'context_encoder_checkpoint_key': 'target_encoder', 'target_encoder_checkpoint_key': 'target_encoder',
        'context_target_share_frozen_ema': True, 'online_encoder_separately_loaded': False,
        'tensors': tensors, 'original_pixel_inputs_read': 0})
    assert encoder_state['norm.weight'].shape == (1024,)
    assert predictor_state['predictor_embed.weight'].shape == (384, 1024)
    assert predictor_state['predictor_proj.weight'].shape == (1024, 384)
    encoder = vision_transformer.vit_large(img_size=(256, 256), patch_size=16, num_frames=64,
        tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False, use_rope=True)
    predictor = predictor_module.vit_predictor(img_size=(256, 256), patch_size=16, num_frames=64,
        tubelet_size=2, embed_dim=1024, predictor_embed_dim=384, out_embed_dim=None,
        depth=12, num_heads=12, num_mask_tokens=10, use_mask_tokens=True,
        use_rope=True, uniform_power=False, use_sdpa=True, use_silu=False, wide_silu=True)
    omitted_encoder = load_strict_with_known_static_position(encoder, encoder_state, 'pos_embed')
    omitted_predictor = load_strict_with_known_static_position(predictor, predictor_state, 'predictor_pos_embed')
    encoder = encoder.float().eval().requires_grad_(False)
    predictor = predictor.float().eval().requires_grad_(False)
    before = {'encoder': state_sha(encoder.state_dict()), 'predictor': state_sha(predictor.state_dict())}
    # Native 16x16 spatial token grid, four synthetic frames = two complete tubelets.
    # Dynamic frame count is supported by the frozen encoder; positions retain their native grid.
    video = torch.linspace(-1, 1, 3*4*256*256).reshape(1, 3, 4, 256, 256)
    context_ids = torch.arange(32, dtype=torch.long).unsqueeze(0)
    target_ids = torch.arange(256, 272, dtype=torch.long).unsqueeze(0)
    block_shapes = []
    handle = encoder.blocks[0].register_forward_pre_hook(lambda module, args: block_shapes.append(list(args[0].shape)))
    with torch.inference_mode():
        context = encoder(video, masks=[context_ids])
        prediction = predictor(context, masks_x=[context_ids], masks_y=[target_ids], mask_index=0)
        changed_only_withheld_tubelet = video.clone()
        changed_only_withheld_tubelet[:, :, 2:] += .12345
        context_changed = encoder(changed_only_withheld_tubelet, masks=[context_ids])
        predicted_changed = predictor(context_changed, masks_x=[context_ids], masks_y=[target_ids], mask_index=0)
        handle.remove()
        target_full = encoder(video)
        target_observed = target_full.gather(1, target_ids.unsqueeze(-1).expand(-1, -1, 1024))
        synthetic_residual = prediction-target_observed
    assert context.shape == (1, 32, 1024)
    assert prediction.shape == target_observed.shape == synthetic_residual.shape == (1, 16, 1024)
    assert target_full.shape == (1, 512, 1024)
    assert all(bool(torch.isfinite(value).all()) for value in [context, prediction, target_observed, synthetic_residual])
    assert block_shapes == [[1, 32, 1024], [1, 32, 1024]], block_shapes
    assert torch.equal(context, context_changed) and torch.equal(prediction, predicted_changed)
    after = {'encoder': state_sha(encoder.state_dict()), 'predictor': state_sha(predictor.state_dict())}
    assert before == after
    assert file_sha(checkpoint) == SHA
    for name, expected in frozen['files'].items():
        assert file_sha(upstream/name) == expected, name
    report = {'status': 'passed', 'asset_sha256': SHA, 'asset_bytes': SIZE, 'device': 'cpu',
        'torch_version': torch.__version__, 'selected_encoder_key': 'target_encoder',
        'context_encoder_checkpoint_key': 'target_encoder', 'target_encoder_checkpoint_key': 'target_encoder',
        'context_target_share_frozen_ema': True, 'official_hub_inference_semantics': True,
        'online_encoder_separately_loaded': False,
        'training_pairing_note': 'Official hub uses EMA for context too. This verifies that inference interface, not online-context/EMA-target training semantics or risk calibration.',
        'strict_encoder_keys': len(encoder.state_dict()), 'strict_predictor_keys': len(predictor.state_dict()),
        'encoder_parameters': sum(value.numel() for value in encoder.parameters()),
        'predictor_parameters': sum(value.numel() for value in predictor.parameters()),
        'known_static_position_tensors_omitted': {'encoder': omitted_encoder, 'predictor': omitted_predictor},
        'input_shape': list(video.shape), 'context_ids_shape': list(context_ids.shape),
        'target_ids_shape': list(target_ids.shape), 'context_shape': list(context.shape),
        'prediction_shape': list(prediction.shape), 'target_observed_shape': list(target_observed.shape),
        'synthetic_residual_shape': list(synthetic_residual.shape), 'finite_verified': True,
        'pre_attention_mask_first_block_shapes': block_shapes,
        'withheld_tubelet_pixel_intervention_exact_context_invariance': True,
        'withheld_tubelet_pixel_intervention_exact_prediction_invariance': True,
        'state_before': before, 'state_after': after, 'states_unchanged': True,
        'checkpoint_and_upstream_unchanged': True, 'model_hub_executed': False,
        'npu_used': False, 'dataset_pixels_read': 0, 'gt_pixels_read': 0, 'quality_scores_computed': False,
        'semantic_scope': 'NEW V-JEPA2-L matched critic asset, not legacy distilled2.1 restorer conditioning',
        'elapsed_seconds': time.monotonic()-began,
        'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(ROOT/'evidence/synthetic_interface.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
