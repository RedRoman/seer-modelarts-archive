# Canonical RGB FIT diagnostic evidence

This is the fixed seven-condition diagnostic on the pre-existing source FIT split: 102 clips / 51 scenes. The 17 internal scenes and target data were not opened or scored. HVI was re-evaluated with canonical RGB input; its output remains RGB. Legacy source GT was explicitly permuted from BGR to RGB once. The original Seer-native RAFT ±2 flow and masks were reused unchanged, so this does not measure newly estimated RGB-HVI correspondences.

The frozen runner, tests and protocol are in the parent phase. `source_manifest.json` records the five exact source/dependency identities used at runtime. All 10 synthetic CPU tests passed without skips. `fit_launch_manifest.json` bound the complete new HVI manifest and export verification before any quality scores were computed. `fit_run_complete.json` records a successful CPU4 run and unchanged sources.

`fit_independent_verification_v2.json` is the final independent arithmetic/provenance check: 2,170 scalar checks, maximum absolute difference 3.55e-15. It additionally pins the old diagnostic result SHA before comparing unchanged geometric support coverage. The earlier verifier and its passing result are retained unchanged; their arithmetic is identical. No prediction was rerun to produce the second verification.

The complete seven-row scores and 102 per-clip / 51 per-scene records are at `../../results/fit_rgb_v1/fit_results.json`. All three same-frame controls exactly equal HVI. The three fixed temporal strengths are reported together; no strength or checkpoint was selected. These are source FIT controls, not a learned-module improvement, generalization result, or SOTA comparison.

The exclusive SFS backup root and every copied file SHA are in `backup_manifest.json`. `download_verification.json` confirms all 23 archived text-file identities after download. The compressed transport archive and original frozen release remain outside Git; no tensor arrays or model weights are included here.
