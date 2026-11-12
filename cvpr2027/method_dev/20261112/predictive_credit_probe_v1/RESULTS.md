# Predictive-credit source FIT12 probe

2026-11-12. Operational experiment record; no manuscript work. The frozen question, actions, controls and decision rule are in [PLAN.md](PLAN.md). No learned parameters, threshold search, internal17 evaluation or target-domain input is part of this run.

## Fixed RGB action: completed

Twelve predeclared FIT scenes, one first-pass clip per scene, central frames8,9 only; one scene contributes256 patch rows. These numbers are **mean scene PSNR computed from each scene's central-two-frame RGB MSE**, not full16-frame clip PSNR and not the previous FIT102 result.

| Fixed action | Mean RGB PSNR | Change from HVI |
|---|---:|---:|
| RGB HVI anchor | 19.008205 | 0 |
| Donor1 DC-preserving action | 19.220426 | +0.212220 |
| Donor2 DC-preserving action | 19.200360 | +0.192155 |
| Pair DC-preserving action | 19.260180 | +0.251975 |
| Same-frame zero action | 19.008205 | exactly0 |
| Existing temporal mean, no DC constraint | 19.512881 | +0.504676 |

The DC pair improves all12 scene PSNRs, with changes from +0.005770 to +0.670932dB. The no-DC temporal mean improves the macro average more, but declines in4/12 scenes. This describes a tradeoff in this small source development subset; it does not show a learned module or establish generalization.

All3072 patches satisfy both donor valid fractions>=.5. Under the registered normalized-pair action recipe, donor1 conditional utility is positive/negative on2622/450 patches and donor2 on2538/534, with no ties within1e-10. Thus5160 donor actions are beneficial and984 harmful. These utility labels refer to the exact singleton-versus-averaged-pair actions; they are not physical causal effects of a frame.

The prediction interface has no GT argument. The GT-free observable sidecar was frozen before label scoring; both-stage observable features match exactly. All five utility identities recompute with zero error from saved scalar MSEs. Same-frame MSE equals anchor MSE exactly. Maximum absolute per-patch/frame/channel DC-sum error is7.56979e-6, below the predeclared2e-5 threshold.

Evidence: [value result](results/values_v1/values_results.json), [independent value replay](checks/values_local_verification.json), [independent action-table replay](checks/independent_action_table_replay.json), [GT-free sidecar](checks/values_evidence_v1/values_observables_v1/observable_sidecar.json), and [SFS receipt](checks/values_backup_receipt.json). Scalar NPZ is outside Git; its SHA256 is `4ccd7e9a807f43a3c64d6b102f3775a2aca9fed86d197643a4e1909c8d338a2c`.

## Predictive-credit gate

**FAIL. Stop this candidate before module training.** All12 scenes remain in the common intersection, all3072 patches are eligible, and all6144 donor labels are nonzero. No scene, donor, score direction, threshold or parameter was selected after the results.

| Score, higher means more useful | Equal-scene macro AUROC |
|---|---:|
| Trained predictor conditional credit, candidate | **0.456270** |
| Random predictor conditional credit | 0.498163 |
| Wrong-time trained predictor conditional credit | 0.480935 |
| Flow-aligned full-context EMA cosine | 0.494526 |
| Negative LQ photometric MSE | **0.562261** |
| Negative HVI consistency MSE | 0.500180 |

The candidate misses the predeclared.55 threshold and fails all five superiority comparisons. Candidate minus LQ consistency is -.105992 AUROC, with paired complete-scene bootstrap95% CI[-.166133,-.044401] (20,000 draws, seed20261123). It also underperforms static EMA cosine with CI[-.066400,-.010297]. The other three intervals include zero; their lower point estimates do not establish significant inferiority. This is a source FIT screen for the exact fixed action recipe, not a statement that all predictive priors are useless.

The tau1 budget diagnostics further limit the module claim: v12 ranges from-.480733 to.431370, so every patch lies below1 and the allocation proposal has **no mathematical freedom on100% of patches**. Budget is zero on51.5625%; mean anchor weight is.98331054. No total or individual capacity reaches its numeric upper bound. Existing Shapley mixing plus caps/softmax is therefore not an independently supported new solver/module mechanism here.

The frozen result is [source_fit_probe_analysis.json](results/analysis_v1/source_fit_probe_analysis.json), SHA256 `f7edb2fb98af3977a05cba7baa999a4cf8f30694e632e4a0ddf0677ff25a44f4`. [Budget diagnostics](results/analysis_v1/budget_summary.json) and [algebra](checks/analysis_budget_algebra_v1.md) separate numerical clipping from an active allocation constraint. The independent root replay uses direct positive-negative pairwise wins/half-ties rather than the analyzer's midrank algorithm: **326 checks pass**, including all72 per-scene AUROCs, scalar utility identities, input hashes, exact grid coverage, macro aggregation and paired means. See [remote replay](checks/root_independent_verification.json) and [local replay after transfer](checks/root_local_independent_verification.json); their receipt bytes match exactly. The verifier imports neither analysis code nor Torch and opens no GT/tensor archive.

Keep fixed temporal/DC fusion as comparison baselines. Do not train or expand this predictive-credit candidate, reverse its score, choose a favourable subset/tau, or attribute the +.251975dB fixed-action gain to it. Current evidence is insufficient for the requested module innovation or SOTA methodology claim. This run supplies no new cross-domain evidence and does not start manuscript work.

## Runtime and provenance

The NPU path did not pass the fixed CPU/NPU normalized-loss parity thresholds. Default precision differed by max.447345/mean.131403; explicit FP32 still differed by trained max.02572/mean.005173 and random max.005362/mean.001426. One intermediate attempt stopped before forward because of a serialization/API error. No NPU dataset forward was accepted. Thresholds were not relaxed.

Formal data execution uses the unchanged official attention and weights on CPU: four independent workers, four threads each, three fixed clips each. Each worker first passes a full16-frame synthetic test, allcoalition/control interfaces, and exact masked context/prediction invariance. A scheduling mkdir bug was found before any dataset access, retained as failed evidence, and repaired in a new release. Eight independent synthetic merge fixtures check completion, identity, state, hash and coverage rejection. All earlier releases and failures remain recorded.

All four full16-frame synthetic results are identical, and all four workers' before/after states are identical. The full probe, including controls and static diagnostics, costs220.5–237.8CPU seconds per source clip (mean228.4, four threads). This is not measured core-module latency. Its merged export passes36,864 finite-loss and108 identity/shape checks. The final process check finds no owned experiment process; NPU6/7 are idle. See [critic closure](checks/critic_export_summary.json) and [runtime notes](checks/critic_runtime_notes.md).

The statistical implementation and gate were frozen before reading actual critic losses or patch utility arrays. Before its final freeze, its author received six descriptive fixed-action macro PSNR values in a message; [the disclosure](checks/analysis_freeze_disclosure_v1.json) records this. These values were not used to change the registered primary rule. Later freeze versions add CPU execution-source bindings only. The protocol/source/fixed-action evidence was committed as `9301fbf332487756e8352437eaba3ccbb51bb786` before actual critic source results were available.

Verification comprises8 critic CPU contract tests,10 value contract tests,27 statistical/guard tests,8 independent merge fixtures, four full-size actual-model synthetic checks, and the independent326-check replay on both remote and local scalar files. Component SFS snapshots separately verify23 value,103 critic and36 analysis payloads. The [root closure manifest](checks/root_closure_manifest.json) inventories final UTF-8 evidence and references those component snapshots; [its receipt](checks/root_final_backup_receipt.json) records the additional exclusive SFS copy of the smaller root closure payload. No image/GT pixels or model weights enter Git.
