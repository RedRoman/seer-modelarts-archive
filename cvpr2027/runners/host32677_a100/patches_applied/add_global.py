import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/token_sufficiency.py"
s = open(p).read()

s = s.replace(
    '''TRANSFORMS = {''',
    '''# A fixed basis estimated once on the train split, as opposed to each clip's
# own principal directions. Only this version defines a code a small network
# could be asked to predict, so Amendment 18 makes it the gate on E27. The
# train-split variance explained by 4 global components is 0.27-0.37, so the
# per-clip result is not expected to survive here, and measuring it is the
# point rather than a formality.
GLOBAL = {"mean": None, "comp": None}


def load_basis(path, dev):
    b = json.load(open(path))
    GLOBAL["mean"] = [torch.tensor(m, device=dev).float() for m in b["mean"]]
    GLOBAL["comp"] = [torch.tensor(c, device=dev).float() for c in b["components"]]
    return b


def global_rank(z, k, li):
    mu = GLOBAL["mean"][li]
    V = GLOBAL["comp"][li][:k]
    b, n, d = z.shape
    x = (z.reshape(-1, d) - mu)
    return ((x @ V.T) @ V + mu).reshape(b, n, d)


TRANSFORMS = {''', 1)

s = s.replace(
    '''        for tname, fn in TRANSFORMS.items():
            zp = [fn(z) for z in zl]''',
    '''        work = dict(TRANSFORMS)
        if GLOBAL["comp"] is not None:
            for k in (4, 16, 64):
                work[f"global rank {k}"] = k      # marker, applied per layer
        for tname, fn in work.items():
            if isinstance(fn, int):
                zp = [global_rank(z, fn, li) for li, z in enumerate(zl)]
            else:
                zp = [fn(z) for z in zl]''', 1)

s = s.replace(
    '''    ap.add_argument("--out", default="")''',
    '''    ap.add_argument("--basis", default="",
                    help="fixed basis from tools/global_basis.py; adds the "
                         "global-rank transforms Amendment 18 gates E27 on")
    ap.add_argument("--out", default="")''', 1)

s = s.replace(
    '''    acc = {k: {"psnr": [], "lpips": []} for k in TRANSFORMS}''',
    '''    if a.basis:
        load_basis(a.basis, dev)
    keys = list(TRANSFORMS) + ([f"global rank {k}" for k in (4, 16, 64)]
                               if a.basis else [])
    acc = {k: {"psnr": [], "lpips": []} for k in keys}''', 1)

s = s.replace('for t in TRANSFORMS:', 'for t in keys:')
s = s.replace('"per_clip": {t: acc[t] for t in TRANSFORMS}',
              '"per_clip": {t: acc[t] for t in keys}')

for probe in ("def global_rank", "--basis", "global rank 4"):
    assert probe in s, probe
ast.parse(s); open(p, "w").write(s); print("patched")
