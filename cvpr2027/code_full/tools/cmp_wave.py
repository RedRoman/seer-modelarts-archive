"""E2 at convergence, and H16' -- both against the three-seed full-token baseline."""
import json, glob, numpy as np

def seqs(f): return {r["name"]: r for r in json.load(open(f))["per_sequence"]}
def boot(d, n=10000):
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(d), size=(n, len(d)))
    return np.percentile(d[i].mean(1), [2.5, 97.5])

B = [f"/Users/hezihao/Documents/Seer_ICLR27/results/cvpr/indomain/{p}" for p in
     ("e17_small_pretrained_s0.json", "e17_small_pretrained_s1.json",
      "e19_small_pretrained_s2.json")]

def cmp(label, vfiles, bfiles=B):
    V = [seqs(f) for f in vfiles]; Bs = [seqs(f) for f in bfiles]
    names = sorted(set.intersection(*[set(x) for x in V + Bs]))
    print("%s  (n=%d, %d vs %d seeds)" % (label, len(names), len(V), len(Bs)))
    for m, nd in (("psnr_y", 3), ("lpips", 4)):
        v = np.array([[x[n][m] for x in V] for n in names]).mean(1)
        b = np.array([[x[n][m] for x in Bs] for n in names]).mean(1)
        d = v - b; ci = boot(d)
        per = [np.mean([x[n][m] for n in names]) - b.mean() for x in V]
        print("  %-7s %8.4f  base %8.4f  d %+8.4f [%+.4f, %+.4f]%s  per-seed %s" % (
            m, v.mean(), b.mean(), d.mean(), ci[0], ci[1],
            "*" if ci[0]*ci[1] > 0 else " ",
            " ".join("%+.3f" % x for x in per)))
    print()

cmp("E2 at convergence: taps 23,23,23,23 vs 5,11,17,23",
    sorted(glob.glob("/Users/hezihao/Documents/Seer_ICLR27/results/cvpr/wave/e2c_deep_s*.json")))
cmp("H16' (ii): corruption-trained gate, in-domain cost",
    sorted(glob.glob("/Users/hezihao/Documents/Seer_ICLR27/results/cvpr/wave/e25_cgate_s[01].json")))

print("H16' (i): what random tokens cost each model")
G = [seqs(f) for f in sorted(glob.glob("/Users/hezihao/Documents/Seer_ICLR27/results/cvpr/wave/e25_cgate_s[01].json"))]
R = [seqs(f) for f in sorted(glob.glob("/Users/hezihao/Documents/Seer_ICLR27/results/cvpr/wave/e25_cgate_randtok_s*.json"))]
names = sorted(set.intersection(*[set(x) for x in G + R]))
g = np.array([[x[n]["psnr_y"] for x in G] for n in names]).mean(1)
r = np.array([[x[n]["psnr_y"] for x in R] for n in names]).mean(1)
d = r - g
print("  corruption-trained gate: real %.3f -> random %.3f  = %+.3f dB %s" % (
    g.mean(), r.mean(), d.mean(), "[%+.3f, %+.3f]" % tuple(boot(d))))
print("  E21 gate (no corruption):                          = -4.764 dB")
print("  ungated baseline:                                  = -4.879 dB")
for s in (0, 1):
    a = json.load(open(f"/Users/hezihao/Documents/Seer_ICLR27/results/cvpr/wave/e25_gate_audit_s{s}.json"))["gate_mean"]
    print("  gate value s%d: real %.4f  random %.4f  (init 0.982)" % (
        s, a["real tokens"], a["random tokens"]))
