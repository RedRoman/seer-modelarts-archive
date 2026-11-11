import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/token_sufficiency.py"
s = open(p).read()
s = s.replace(
    '''            pred = out["out"] if isinstance(out, dict) else out
            pred = pred.float().clamp(0, 1)''',
    '''            pred = out["pred_pixels"].float().clamp(0, 1)''', 1)
assert 'out["pred_pixels"]' in s
ast.parse(s); open(p, "w").write(s); print("fixed")
