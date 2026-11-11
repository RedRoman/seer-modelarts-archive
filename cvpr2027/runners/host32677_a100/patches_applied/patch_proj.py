import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/seer_train_cvpr.py"
s = open(p).read(); orig = s

s = s.replace(
    '''    ap.add_argument("--token_corrupt", type=float, default=0.0,''',
    '''    ap.add_argument("--token_proj", choices=["none", "clip_rank", "global_rank"],
                    default="none",
                    help="project the tokens before the branch reads them. "
                         "clip_rank uses each clip's own principal directions, "
                         "which preserves the prior at rank 4 (H20); "
                         "global_rank uses a fixed basis from "
                         "tools/global_basis.py, which needs rank 64 to be "
                         "harmless and destroys the prior at rank 4 (H18).")
    ap.add_argument("--token_proj_k", type=int, default=4)
    ap.add_argument("--token_basis", default="",
                    help="basis JSON, required for global_rank")
    ap.add_argument("--token_corrupt", type=float, default=0.0,''', 1)

s = s.replace(
    '''            zp = [z.float() for z in zl]; l_jep = lq.new_zeros(())''',
    '''            zp = [z.float() for z in zl]; l_jep = lq.new_zeros(())
            if args.token_proj != "none":
                zp = [project_tokens(z, args.token_proj, args.token_proj_k,
                                     basis, li) for li, z in enumerate(zp)]''', 1)

# the projection itself, next to zero_tokens so both token helpers sit together
s = s.replace(
    '''def zero_tokens(b, dev):''',
    '''def project_tokens(z, mode, k, basis, li):
    """Rank-k projection of one tapped layer's tokens.

    clip_rank takes the clip's own principal directions, so the subspace moves
    with the content; global_rank takes a fixed subspace estimated once on the
    training split. The two are measured to differ by 4 dB at rank 4, which is
    the whole reason both are offered rather than one.
    """
    b, n, d = z.shape
    if mode == "global_rank":
        mu, V = basis["mean"][li], basis["comp"][li][:k]
        x = z.reshape(-1, d) - mu
        return ((x @ V.T) @ V + mu).reshape(b, n, d)
    x = z.reshape(-1, d)
    mu = x.mean(0, keepdim=True)
    xc = (x - mu).float()
    V = torch.linalg.svd(xc, full_matrices=False)[2][:k]
    return (((xc @ V.T) @ V) + mu).reshape(b, n, d).to(z.dtype)


def zero_tokens(b, dev):''', 1)

s = s.replace(
    '''    tok_cache, n_corrupt = [None], [0]''',
    '''    tok_cache, n_corrupt = [None], [0]
    basis = None
    if args.token_proj == "global_rank":
        if not args.token_basis:
            raise SystemExit("--token_proj global_rank needs --token_basis")
        bj = json.load(open(args.token_basis))
        if bj["rank"] < args.token_proj_k:
            raise SystemExit(f"basis has rank {bj['rank']} < k={args.token_proj_k}")
        basis = {"mean": [torch.tensor(m, device=dev).float() for m in bj["mean"]],
                 "comp": [torch.tensor(c, device=dev).float() for c in bj["components"]]}
        log.info("  fixed token basis: rank %d of %d available, from %s",
                 args.token_proj_k, bj["rank"], os.path.basename(args.token_basis))
    elif args.token_proj == "clip_rank":
        log.info("  per-clip token projection: rank %d", args.token_proj_k)''', 1)

assert s != orig
for probe in ("def project_tokens", "--token_proj", "basis = None"):
    assert probe in s, probe
ast.parse(s); open(p, "w").write(s); print("patched, syntax ok")
