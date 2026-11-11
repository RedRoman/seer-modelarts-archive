import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/seer_train_cvpr.py"
s = open(p).read(); orig = s

s = s.replace(
    '''    ap.add_argument("--tdelta_form", choices=["relative", "cosine"],''',
    '''    ap.add_argument("--token_corrupt", type=float, default=0.0,
                    help="probability of replacing this step's tokens with the "
                         "previous step's (E25/H16). Real V-JEPA tokens, wrong "
                         "content: the only training condition under which "
                         "closing the gate reduces the loss. 0 disables and "
                         "reproduces every earlier run exactly.")
    ap.add_argument("--tdelta_form", choices=["relative", "cosine"],''', 1)

s = s.replace(
    '''            zp = [z.float() for z in zl]; l_jep = lq.new_zeros(())''',
    '''            zp = [z.float() for z in zl]; l_jep = lq.new_zeros(())
            # E25/H16. E21's gate never moved off its initialisation (0.9832
            # with real tokens, 0.9833 with random ones) because it only ever
            # saw correct tokens, so closing never lowered the loss. Here a
            # fraction of steps condition the branch on the previous clip's
            # tokens while the target stays this clip's: in-distribution as
            # tokens, wrong for the content, which is exactly the situation the
            # gate's agreement input is meant to detect. Cached under no_grad
            # already, so the cost is memory, not compute.
            if args.token_corrupt > 0:
                if (tok_cache[0] is not None
                        and tok_cache[0][0].shape[0] == zp[0].shape[0]
                        and random.random() < args.token_corrupt):
                    zp = tok_cache[0]
                    n_corrupt[0] += 1
                else:
                    tok_cache[0] = [z.detach().clone() for z in zp]''', 1)

# allocate the cache next to the other loop state
s = s.replace(
    '''        lq, gt = b["lq"].to(dev, non_blocking=True), b["hq"].to(dev, non_blocking=True)''',
    '''        lq, gt = b["lq"].to(dev, non_blocking=True), b["hq"].to(dev, non_blocking=True)''', 1)

assert "tok_cache[0]" in s
ast.parse(s)
open(p, "w").write(s)
print("step 1 ok")
