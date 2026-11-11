import ast
p = "/cache/zihaohe/seer_cvpr/code/vjepa_llve/tools/dump_qual_cvpr.py"
s = open(p).read()
s = s.replace(
    '''            if tag != "gt":
                rows[name][tag] = {
                    "psnr_y": psnr_y_uint8(f, gt[mid]),
                    "lpips": lpips_of(f[None].astype(np.float32) / 255.,
                                      gt[None, mid].astype(np.float32) / 255., dev)}''',
    '''            if tag != "gt":
                # lpips_of wants (B,3,H,W) float tensors in [0,1], the same
                # form eval_e4 feeds it, so the figure's numbers and the
                # table's come from identical inputs
                def to_t(u8):
                    return (torch.from_numpy(np.ascontiguousarray(u8[None]))
                            .permute(0, 3, 1, 2).float().div(255.))
                rows[name][tag] = {
                    "psnr_y": psnr_y_uint8(f, gt[mid]),
                    "lpips": lpips_of(to_t(f), to_t(gt[mid]), dev)}''', 1)
assert "def to_t(u8):" in s
ast.parse(s); open(p, "w").write(s); print("fixed")
