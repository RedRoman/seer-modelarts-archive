import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/eval_e4.py"
s = open(p).read(); orig = s

s = s.replace(
    "def build_model(ckpt, dev):",
    "def build_model(ckpt, dev, force_arm=None):", 1)

# the conditioning source is the only thing force_arm changes; weights, shapes
# and the exposure head all stay as trained
s = s.replace(
    '''    arm = saved.get("vjepa", "pretrained")''',
    '''    arm = saved.get("vjepa", "pretrained")
    trained_arm = arm
    if force_arm is not None and force_arm != arm:
        # H16 criterion (ii) is an inference-only test: take a model trained
        # with the real tokens and feed it tokens from a randomly initialised
        # encoder. Nothing but the token source changes, so a trained gate that
        # has learned to recognise unusable conditioning should close and
        # recover the no-token result rather than losing the 4.1 dB that the
        # ungated model loses.
        if force_arm == "none":
            raise ValueError("force_arm=none would leave the cross-attention "
                             "without keys; it is a different architecture")
        arm = force_arm''', 1)

s = s.replace(
    '''    ap.add_argument("--limit", type=int, default=None)''',
    '''    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force_arm", default=None, choices=["pretrained", "random"],
                    help="score a trained checkpoint with a different token "
                         "source than it was trained with (H16 criterion ii)")''', 1)

s = s.replace("build_model(args.ckpt, dev)",
              "build_model(args.ckpt, dev, force_arm=args.force_arm)")

assert s != orig
for probe in ("force_arm=None", "--force_arm", "trained_arm = arm"):
    assert probe in s, probe
ast.parse(s)
open(p, "w").write(s)
print("patched, syntax ok")
