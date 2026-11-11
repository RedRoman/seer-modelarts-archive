import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/token_sufficiency.py"
s = open(p).read()
s = s.replace(
    '''        json.dump({"ckpt": a.ckpt, "clips": a.clips, "result": res},
                  open(a.out, "w"), indent=1)''',
    '''        # per-clip values so the differences can be paired and bootstrapped;
        # a mean on its own is the kind of number this project has been told
        # not to report
        json.dump({"ckpt": a.ckpt, "clips": a.clips, "result": res,
                   "per_clip": {t: acc[t] for t in TRANSFORMS}},
                  open(a.out, "w"), indent=1)''', 1)
assert '"per_clip"' in s
ast.parse(s); open(p, "w").write(s); print("patched")
