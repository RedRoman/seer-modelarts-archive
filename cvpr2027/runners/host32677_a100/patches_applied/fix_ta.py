import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/temporal_ablate.py"
s = open(p).read()
s = s.replace(
    """            for arm, (pb, bb, refiner, a, clip_len, _) in models.items():
                pred = run_clip(pb, bb, refiner, a, lq_in, clip_len, dev, amp)""",
    """            # build_model returns (pb, bb, refiner, exp_head, arm, clip_len,
            # saved); unpack by name so a further addition fails loudly at the
            # call rather than silently shifting an argument
            for arm, m in models.items():
                pb, bb, refiner, exp_head, a, clip_len = m[0], m[1], m[2], m[3], m[4], m[5]
                pred = run_clip(pb, bb, refiner, exp_head, a, lq_in, clip_len,
                                dev, amp)""", 1)
assert "m[0], m[1], m[2], m[3], m[4], m[5]" in s
ast.parse(s); open(p, "w").write(s); print("fixed")
