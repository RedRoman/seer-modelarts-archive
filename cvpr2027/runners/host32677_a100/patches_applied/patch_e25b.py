import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/seer_train_cvpr.py"
s = open(p).read()

s = s.replace("import os\nimport sys", "import os\nimport random\nimport sys", 1)

s = s.replace(
    '''    it = accum = 0
    run = {"total": 0.0, "pix": 0.0, "jepa": 0.0, "exp": 0.0, "td": 0.0}''',
    '''    it = accum = 0
    run = {"total": 0.0, "pix": 0.0, "jepa": 0.0, "exp": 0.0, "td": 0.0}
    # one-element lists so the training loop can rebind them without `nonlocal`
    tok_cache, n_corrupt = [None], [0]
    if args.token_corrupt > 0:
        random.seed(args.seed)
        log.info("  token corruption on: p=%.2f, previous step's tokens",
                 args.token_corrupt)''', 1)

# surface the realised corruption rate, so "configured" is distinguishable
# from "fired" in the log rather than assumed
s = s.replace(
    '''f"jepa={run['jepa']/n:.4f} exp={run['exp']/n:.4f} td={run['td']/n:.4f} gate={g:+.5f} "''',
    '''f"jepa={run['jepa']/n:.4f} exp={run['exp']/n:.4f} td={run['td']/n:.4f} "
                         f"corrupt={n_corrupt[0]} gate={g:+.5f} "''', 1)

for probe in ("import random", "tok_cache, n_corrupt = [None], [0]", "corrupt={n_corrupt[0]}"):
    assert probe in s, probe
ast.parse(s)
open(p, "w").write(s)
print("patched, syntax ok")
