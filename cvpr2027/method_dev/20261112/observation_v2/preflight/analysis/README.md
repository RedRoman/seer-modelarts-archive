Observation-v2 analysis freeze — before real v2 outcomes

This folder freezes the independent NumPy-only analyzer and synthetic validation. No real v2 result has been analyzed or generated here. Synthetic fixtures exist only in temporary test directories and are explicitly named SYNTHETIC. No model, Torch module, frame, tensor cache, GT image or checkpoint is loaded by the analyzer.

The analyzer requires both full pretrained and no-token VAL16 JSON files, all 31 methods in each, all 21 new heads and three frozen v1 raw/direct references, and the frozen source/split/code provenance. It verifies reported means, absolute per-sequence tLP gap, common GT references, complete source/target pairing, fit-only scope, new clip-first sample-plan SHAs, initialization matching and nested adjacent-cache lineage. Source code checks use the immutable runtime source tree passed with `--code`; if repository code later changes, use the corresponding frozen release rather than weakening verification.

All 12 declared within-arm contrasts are retained for both arms. The output additionally retains all 31 individual methods, 15 fixed/seed-averaged method groups, per-seed and per-scene effects, 15 absolute arm comparisons and the 12 differences of within-arm effects. Fixed methods repeated for seed pairing are explicitly flagged; they are not extra independent replications. All differences and raw CIs remain left-minus-right. Only the nested `benefit_oriented` fields reverse LPIPS/tLP-gap sign and reverse interval endpoints, so positive there consistently means favorable.

The numerical order matches the prior analysis: first average the three paired head-seed effects within each sequence, then the two conditions within each scene. Generate one `np.random.default_rng(0).integers(0,8,size=(10000,8))` draw matrix and reuse it for all arms, metrics and contrasts. These are unadjusted exploratory percentile95 intervals conditional on three already fitted heads and one historical base per arm, with uncertainty resampled only over eight scenes. They neither replicate base training nor adjust for multiple comparisons. Historical base pixel initializations/training histories differ; the arm interaction cannot isolate a causal video-prior effect. No SOTA threshold, model selection, loss change or novelty conclusion is encoded.

Example, supplying actual result paths only after the complete evaluations finish:

```sh
python3 analyze_observation_v2.py \
  --pretrained /absolute/path/pretrained_result.json \
  --none /absolute/path/none_result.json \
  --splits '/Users/hezihao/Documents/New project/seer-modelarts-archive/cvpr2027/results/splits_cvpr.json' \
  --code '/absolute/path/to/immutable/release/code_full' \
  --out /absolute/path/to/new_analysis.json
```

Both new_analysis.json and new_analysis.md must be absent. Existing input JSONs are read-only.

Synthetic tests passed 6/6 with no Torch installed, including an independently constructed full two-arm metadata fixture; exact legacy bootstrap draw/order outputs; scene rather than sequence resampling; seed pairing; raw and benefit-oriented signs; actual pure source-plan/control-recipe reconstruction; and negative cases for coverage, seeds, initialization, plan, mean, temporal gap, target pairing and GT-reference mismatches. The statistics core was independently read-only reviewed by the architecture agent, which reported no blocker. Test log is synthetic_tests.log.

To rerun the frozen tests against the matching source tree:

```sh
SEER_ANALYSIS_CVPR='/Users/hezihao/Documents/New project/seer-modelarts-archive/cvpr2027' \
  python3 -m unittest discover -s . -p test_analyze_observation_v2.py -v
```

The sibling reference_analyze_source_transfer.py is a byte-identical read-only copy of the prior auditor, SHA256 1e61d95af69a5be89aef35ccb03fbf0af73df69bac6c9e807fa62fb85d33cbc7. The analyzer pins this SHA and never modifies the reference or any frozen v1 helper/entry point.
