import re, sys
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/seer_train_cvpr.py"
s = open(p).read()
orig = s

s = s.replace(
    "from losses.temporal_delta import temporal_delta_loss  # noqa: E402",
    "from losses.temporal_delta import (temporal_delta_loss,  # noqa: E402\n"
    "                                   temporal_delta_cosine_loss)")

s = s.replace(
    '''    ap.add_argument("--layers", choices=["single", "all"], default="single",''',
    '''    ap.add_argument("--tdelta_form", choices=["relative", "cosine"],
                    default="relative",
                    help="relative is E22's |dp-dg|/|dg|, which a prediction "
                         "with no temporal variation satisfies at 1.0; cosine "
                         "is E24's direction term plus a symmetric log-ratio "
                         "magnitude penalty, which shrinking cannot satisfy")
    ap.add_argument("--layers", choices=["single", "all"], default="single",''')

s = s.replace(
    '''            l_td = temporal_delta_loss(zp_hat.float(), zp_gt.float(),
                                       (args.clip_len // 2, 16, 16))''',
    '''            td_fn = (temporal_delta_cosine_loss
                     if args.tdelta_form == "cosine" else temporal_delta_loss)
            l_td = td_fn(zp_hat.float(), zp_gt.float(),
                         (args.clip_len // 2, 16, 16))''')

assert s != orig, "no replacement applied"
for probe in ("temporal_delta_cosine_loss", "--tdelta_form", "td_fn("):
    assert probe in s, probe
open(p, "w").write(s)
import ast; ast.parse(s)
print("patched, syntax ok")
