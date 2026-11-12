"""Frozen author-labelled DID TempRetinex reference on BVI VAL16 only.

Not a matched SDSD-only or SOTA comparison: the release does not document its
exact training split/run. Pinned official inference definitions are executed
in an isolated namespace; training imports and device-specific loading are
excluded. Learned enhancement keys and the separate released RAFT state load
strictly. All forward operations, including Denoise_2's four-view ensemble,
prediction-history histogram matching and RAFT's 20 iterations, are retained.

We use the common centre16/crop256 protocol instead of upstream full-sequence
1920x1080 resize, reset recurrence at each clip, run forward only, and round
RGB to uint8 for common scoring (upstream save_images truncates). No reverse
pass, target optimization, GT histogram matching or GT brightness adjustment.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
COMMIT = "03ca79f57f2ad7b4c6120364de3a8c59afa42eed"
LABEL = "TempRetinex author-labelled DID released reference; frozen BVI VAL16"
SOURCE_SHA256 = {
    "model/model.py": "f1dbd2a7ba1713ee3f09be1d8893683930707432e4e4ac9150594a6d2f61eb82",
    "utils/utils.py": "9a0d8be54139f72818b06b8dddc363d8094bbef025fd6c3d1b331a9d726d09e8",
    "model/RAFT/extractor.py": "49ef9555b0d7e1163fd3e9f754e02b99d68ed866cbacff45c3ddec1748b43d79",
    "model/RAFT/update.py": "7302f91ffc24c85cf739a25feb17d9d1e537f7344a2ddc331daf8ab9db6c3f74",
    "model/RAFT/corr.py": "c99a4c478d99ae97b3681479debc74b8d8999efa47091afb3836fcc943eb570d",
    "model/RAFT/raft.py": "76c42b5f37615315414a21be67a4a3f6891585cdb0599e7aaf9b507c0a769099",
}
WEIGHT_SHA256 = {
    "weights/DID.pth": "bc0183460105ac3a28205ca13fe211147f64bd8e5bbf564b41a56bfa336f2b5a",
    "weights/raft-sintel-finetuned.pth": "be03481a61bc67a3b7907e6d2c15aae6514864269195deef933c82ff992551d5",
}
HELPER_SHA256 = {
    "tools/eval_stcd_transfer.py": "420fdc33046c90bfda486df3162336f1da56c52562c719d41be102c7bbf63545",
    "tools/eval_e4.py": "3fb906535d76ca11e379d41113424d1d1d47205d56ca5f5de0b557c3adcf0467",
    "data/common.py": "6041b43e832ab67def75b9e7285927dddd77d45d0bc25945bb600167b04e4ded",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_files(root, expected):
    for relative, sha in expected.items():
        if digest(Path(root) / relative) != sha:
            raise ValueError(f"Pinned file mismatch: {relative}")


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def official_namespace(root):
    """Execute only named definitions from SHA-pinned files, never their CLI."""
    import torch
    check_files(root, SOURCE_SHA256)
    ns = {'__name__': '_seer_pinned_tempretinex', 'torch': torch,
          'nn': torch.nn, 'F': torch.nn.functional, 'np': np,
          'autocast': torch.cuda.amp.autocast}
    selections = {
        'utils/utils.py': ['warp_tensor', 'histogram_match_tensor', 'calc_alpha',
                           'coords_grid', 'upflow8', 'bilinear_sampler'],
        'model/RAFT/extractor.py': ['ResidualBlock', 'BottleneckBlock', 'BasicEncoder', 'SmallEncoder'],
        'model/RAFT/update.py': ['FlowHead', 'ConvGRU', 'SepConvGRU', 'SmallMotionEncoder',
                               'BasicMotionEncoder', 'SmallUpdateBlock', 'BasicUpdateBlock'],
        'model/RAFT/corr.py': ['CorrBlock'],
        'model/RAFT/raft.py': ['RAFT'],
        'model/model.py': ['Denoise_1', 'Denoise_2', 'Enhancer', 'Finetunemodel'],
    }
    selected_hashes = {}
    for relative, names in selections.items():
        source = (Path(root) / relative).read_text()
        tree = ast.parse(source, filename=relative)
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in names]
        if [node.name for node in nodes] != names:
            # Source order is not required, but every named definition is unique.
            if len(nodes) != len(names) or {node.name for node in nodes} != set(names):
                raise ValueError(f"Unexpected pinned definitions: {relative}")
        selected_hashes[relative] = hashlib.sha256(
            '\n'.join(ast.get_source_segment(source, node) for node in nodes).encode()).hexdigest()
        exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), ns)
    return ns, selected_hashes


def load_model(root, device):
    import torch
    root = Path(root)
    check_files(root, WEIGHT_SHA256)
    ns, selected_hashes = official_namespace(root)
    release = torch.load(root / 'weights/DID.pth', map_location='cpu', weights_only=True)
    if not isinstance(release, dict) or set(release) != {'epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'}:
        raise ValueError('Unexpected DID checkpoint structure')
    state = release['model_state_dict']
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected a complete tensor state dictionary')
    # Equivalent inference constructor, replacing permissive filtering/CUDA load.
    model = ns['Finetunemodel'].__new__(ns['Finetunemodel'])
    torch.nn.Module.__init__(model)
    model.enhance = ns['Enhancer'](layers=3, channels=64)
    model.denoise_1 = ns['Denoise_1'](chan_embed=48)
    model.denoise_2 = ns['Denoise_2'](chan_embed=48)
    enhancement = {k: v for k, v in state.items() if not k.startswith('raft.')}
    model.load_state_dict(enhancement, strict=True)
    # Shared module aliases must also agree, rather than accepting last-key wins.
    if any(not torch.equal(v.cpu(), enhancement[k].cpu()) for k, v in model.state_dict().items()):
        raise ValueError('Loaded enhancement values differ from released keys')
    args = argparse.Namespace(of_scale=1)
    raft = ns['RAFT'](args)
    flow = torch.load(root / 'weights/raft-sintel-finetuned.pth', map_location='cpu', weights_only=True)
    if not isinstance(flow, dict) or not flow or not all(k.startswith('module.') and isinstance(v, torch.Tensor) for k, v in flow.items()):
        raise ValueError('Expected released DataParallel RAFT tensor dictionary')
    flow = {k[7:]: v for k, v in flow.items()}
    raft.load_state_dict(flow, strict=True)
    embedded_flow = {k[5:]: v for k, v in state.items() if k.startswith('raft.')}
    if set(embedded_flow) != set(flow):
        raise ValueError('Unexpected embedded RAFT keys in DID checkpoint')
    embedded_equal = all(torch.equal(embedded_flow[k], flow[k]) for k in flow)
    model.raft = raft
    model.of_scale = 1
    for name in ('last_H3', 'last_H3_wp', 'last_s3', 'last_s3_wp'):
        setattr(model, name, None)
    model.is_new_seq = True
    model = model.float().requires_grad_(False).to(device).eval()
    p = {
        'label': LABEL, 'repo': 'https://github.com/liyinibristol/TempRetinex', 'commit': COMMIT,
        'source_root': str(root.resolve()), 'upstream_source_sha256': SOURCE_SHA256,
        'executed_definition_sha256': selected_hashes, 'weight_sha256': WEIGHT_SHA256,
        'weight_urls': {k: f'https://raw.githubusercontent.com/liyinibristol/TempRetinex/{COMMIT}/{k}' for k in WEIGHT_SHA256},
        'checkpoint_epoch': release['epoch'], 'checkpoint_training_loss': release['loss'],
        'training_provenance': 'Author filename DID; exact source split, run config and training data not independently documented; no SDSD-only claim',
        'strict_enhancement_keys': len(enhancement), 'strict_raft_keys': len(flow),
        'embedded_raft_equals_separately_released_raft': embedded_equal,
        'embedded_raft_state_sha256': state_digest(embedded_flow),
        'raft_state_sha256': state_digest(model.raft.state_dict()),
        'state_sha256': state_digest(model.state_dict()),
        'parameters': sum(p.numel() for p in model.parameters()),
        'inference': {'entrypoint_reference': 'official predict.py; its of_scale default is 1 (test.py default is 3)',
                      'of_scale': 1, 'raft_iterations': 20, 'mixed_precision': False,
                      'precision': 'float32', 'frame_order': 'ascending forward only',
                      'reverse_pass': False, 'temporal_ensemble': False,
                      'denoise2_spatial_ensemble': 'official four views: identity, vertical, horizontal, rotation90',
                      'recurrent_reset': 'first frame of each centre16 clip; no preceding full-video context',
                      'internal_histogram_matching': 'current denoised low image to preceding prediction, retained',
                      'gt_histogram_matching': False, 'target_optimization': False,
                      'brightness': 'official low-input calc_alpha, unchanged coefficients/threshold',
                      'output': 'Finetunemodel.forward()[1], H3, RGB'},
        'adaptation': 'Execute named pinned definitions, bypass training/plotting imports; constructor strict-load on CPU then device; all forward methods unchanged',
    }
    return model, p


def enhance_clip(model, low):
    """RGB [T,3,H,W] -> RGB same shape; only LQ enters this boundary."""
    import torch
    if (low.ndim != 4 or low.shape[1] != 3 or len(low) < 2 or min(low.shape[-2:]) < 128
            or any(x % 8 for x in low.shape[-2:]) or not torch.isfinite(low).all()
            or low.min() < 0 or low.max() > 1):
        raise ValueError('Expected finite RGB [0,1], T>=2, spatial multiples of 8 and >=128')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Inference requires an eval-mode frozen model')
    values = []
    with torch.no_grad():
        for t, image in enumerate(low):
            model.is_new_seq = t == 0
            result = model(image.unsqueeze(0))
            if not isinstance(result, tuple) or len(result) != 4:
                raise ValueError('Unexpected official inference outputs')
            output = result[1]
            if output.shape != image.unsqueeze(0).shape or not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite or incorrectly shaped enhanced frame')
            values.append(output)
    return torch.cat(values).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--seer-root', type=Path, default=Path(os.environ.get('SEER_ROOT', '/temp/zihaohe/seer_cvpr')))
    parser.add_argument('--split-path', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'npu', 'cuda'], default='npu')
    parser.add_argument('--cpu-smoke', action='store_true', help='Strict load and synthetic two-frame128/recurrent-reset checks; no dataset access')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.cpu_smoke and args.device != 'cpu':
        raise ValueError('--cpu-smoke requires --device cpu')
    os.environ['SEER_ROOT'] = str(args.seer_root)
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(CODE / 'tools'))
    import torch
    if args.device == 'npu':
        import torch_npu  # noqa: F401
    torch.set_num_threads(4)
    device = torch.device('cpu' if args.device == 'cpu' else args.device + ':0')
    started = time.monotonic()
    model, provenance = load_model(args.upstream_root, device)
    if args.cpu_smoke:
        generator = torch.Generator().manual_seed(2)
        low = torch.rand((2, 3, 128, 128), generator=generator) * .1
        output = enhance_clip(model, low)
        # A second clip must discard previous recursive state at its first frame.
        repeated = enhance_clip(model, low)
        torch.testing.assert_close(output, repeated, rtol=0, atol=0)
        if output.shape != low.shape or not torch.isfinite(output).all():
            raise AssertionError('Synthetic recurrent forward failed')
        result = {'purpose': 'synthetic CPU strict-load/forward/reset test; no BVI access',
                  'shape': list(output.shape), 'finite': True, 'clip_reset_exact': True,
                  'output_sha256': hashlib.sha256(output.numpy().tobytes()).hexdigest(),
                  'provenance': provenance}
    else:
        check_files(CODE, HELPER_SHA256)
        from eval_stcd_transfer import validation_inventory, evaluate_sequences, score_uint8
        import eval_e4
        import common.eval_lib as metric_lib
        split = args.split_path or args.seer_root / 'datasets/splits_cvpr.json'
        inventory = validation_inventory(split, args.seer_root)
        # Official reset decision depends on adjacent numeric frame identifiers.
        for row in inventory:
            ids = [int(Path(name).stem.split('.')[-1]) for name in row['selected_frame_names']]
            if any(b != a + 1 for a, b in zip(ids, ids[1:])):
                raise ValueError('Nonconsecutive frame IDs would change official recurrent reset semantics')
        eval_e4.SPLITS = str(split)
        def predict(low_u8):
            low = torch.from_numpy(np.ascontiguousarray(low_u8)).float().permute(0, 3, 1, 2).div_(255).to(device)
            prediction = enhance_clip(model, low)
            return (prediction.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        rows = evaluate_sequences(eval_e4.held_out_sequences('bvi', 'val', 16, 256), predict,
                                  lambda p, g: score_uint8(p, g, device), [r['name'] for r in inventory])
        result = {'method': LABEL, 'domain': 'bvi', 'split': 'val', 'n': len(rows),
                  'protocol': 'common centre16/centre256 uint8 RGB/Y PSNR, LPIPS, consecutive tLP and absolute tLP-GT gap',
                  'upstream_protocol_differences': 'crop256 instead of resize1920x1080; centre16 instead of full sequence; recurrence reset at clip start; round uint8 instead of save_images truncation',
                  'gt_policy': 'LQ-only prediction; GT only for subsequent common metrics; no GT histogram/brightness matching or target optimization',
                  'comparison_scope': 'unmatched author-labelled DID external reference; not SDSD-only/SOTA evidence; checkpoint source details unknown',
                  'per_sequence': rows, 'means': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'name'},
                  'validation_pair_inventory': inventory, 'split_sha256': digest(split),
                  'provenance': provenance, 'scoring_source_sha256': {str(p): digest(p) for p in
                      (Path(__file__), CODE / 'tools/eval_stcd_transfer.py', Path(eval_e4.__file__),
                       CODE / 'data/common.py', Path(metric_lib.__file__))}}
    final_sha = state_digest(model.state_dict())
    final_raft_sha = state_digest(model.raft.state_dict())
    if final_sha != provenance['state_sha256'] or final_raft_sha != provenance['raft_state_sha256']:
        raise AssertionError('Frozen model/RAFT state changed during inference')
    check_files(args.upstream_root, {**SOURCE_SHA256, **WEIGHT_SHA256})
    result['verified_frozen_state_sha256'] = final_sha
    result['verified_frozen_raft_state_sha256'] = final_raft_sha
    result['runtime'] = {'torch': str(torch.__version__), 'device': str(device), 'cpu_threads': 4,
                         'elapsed_seconds': time.monotonic() - started}
    result['adapter_sha256'] = digest(__file__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
