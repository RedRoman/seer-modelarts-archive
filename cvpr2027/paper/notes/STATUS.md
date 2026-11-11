# Seer -> ICLR 2027: working status

Full-paper deadline: 2026-09-26 19:59 Beijing (Sep 25 AoE). Main text <= 9 pages.

## Build
- `tools/sync_results.sh` pulls every eval JSON from the shared disk into `results/seer_iclr_evals/`
- `python3 tools/make_tables.py` regenerates `tables/*.tex`, `tables/result_macros.tex`, `tables/PROVENANCE.md`
- `latexmk -pdf -outdir=build main.tex`
- Any result that has not been measured yet prints a red **TBD**. There are no hand-typed numbers in tables/.

## Result files each table waits for (results/seer_iclr_evals/)
| Table | Files |
|---|---|
| main / temporal / extra / clean6 | `ctrl_vjepa_s{0,1,2}_eval_unified.json`, `ctrl_novjepa_s{0,1,2}_eval_unified.json`, `Retinexformer_union_seersplit_{eval_v2,L0seer_eval_v2}.json` |
| control | above + `abl_randvjepa_s{0,1}_eval_unified.json` |
| ablation | `ctrl_vjepa_s*_step_750_eval_unified.json`, `abl_<name>_s0_eval_unified.json` |
| cp | `ctrl_vjepa_s*_eval_unified.json` (raw / CPfit / CPxfit / archived CP) |
| factorial (not yet in paper) | `Retinexformer_{union,indoor}_seersplit_*`; Seer column from archived runs |

## Changes vs the AAAI version and why
1. Every number is recomputed from per-sequence JSON. All `\pnum` placeholder cells are gone.
2. The L0 LQ/GT pairing bug is fixed: six SDSD-indoor sequences have offset file names, and all baseline L0 numbers were affected.
3. L0 now means Seer's 15-sequence test split for every method, and a clean-6 subset is reported alongside it. Released checkpoints' training-set membership is reported in its own table.
4. The method section now describes what the code actually does:
   - M2 supervises M1 (Psi(Phi(X)) vs Phi(Y)); it is not a loss on Y-hat.
   - M1 acts on layer 23 only.
   - IGAB is used in the decoder only.
   - Cross-attention is applied at scales 2-5.
   - The loss is 1.5*Charbonnier and TV is spatial only.
   - M1 has 0.69M trainable parameters (+22.3M frozen).
5. CP:
   - The archived alpha (0.50, 0.50, 0.68) was not a closed-form fit. It was hand-picked as "Mid" between an SDSD fit (0.41, 0.40, 0.57) and a BVI fit (0.60, 0.57, 0.78), based on BVI test PSNR (`results/aaai_data/per_channel_alpha.json`).
   - The ICLR version reports CP-src (closed-form fit on 15 SDSD training clips; the smoke test on the reference ckpt gave (0.17, 0, 0) after clipping) and CP-tgt (cross-fitted on the target split, folded by scene).
   - "Bayesian" is renamed to closed-form LMMSE, alpha is clipped to [0,1], and the zero-denominator case is handled.
6. The encoder table uses raw L2 (the archived CP alpha was test-tuned). VideoMAE is added to the invariance study, where it comes out more invariant than V-JEPA.
7. New experiments:
   - no-V-JEPA and random-V-JEPA controls
   - component ablations
   - retrained Retinexformer on Seer's split
   - STCD and all baselines under one protocol
   - warping error (RAFT)
   - 3 seeds
8. Added citations: DA-CLIP, Xu et al. 2024, quadruple priors, TCE-Net, RetinexMCNet, RAFT.

## Open decisions for the authors
- Title: 2026-09-25 the user chose to KEEP "A Cross-Domain Prior for Low-Light Video Enhancement". The text must then show what is cross-domain; if the controlled study shows no cross-domain gain, bring this back to the user before submission.
- Archived CP alpha: the user chose to keep it in the appendix with its provenance stated (tab_cp_archived); it supports no claim.
- The claims in the abstract, intro and conclusion wait for the controlled study.
- The co-authors need to be told that the AAAI numbers were not measured.
- [ ] appendix app:cp: interpret CP-src alpha once ctrl evals are in

## Qualitative figure (automated)
- 30313 `qual_when_ready_seer_iclr.sh` waits for `/temp/.../seer_iclr_ckpts/ctrl_{vjepa,novjepa}_s0_final.pt` (staged by `stage_ckpt_seer_iclr.sh` on 30313/30674 after each eval), then dumps to `seer_iclr_evals/qualitative_final/`.
- Local: `scp -r ... qualitative_final results/qualitative/` then `python3 tools/make_fig_qualitative.py --run results/qualitative/qualitative_final` (writes figures/fig_qualitative.pdf; add `--no_cp` if the CP-src column equals raw).
- AAAI figures kept as figures/aaai_*.pdf (originals also in ~/Documents/HMP-Net_AAAI27/figures).

## Job assignment (22:35, static; shared mkdir claims were not exclusive on the SFS share)
| host | now | next (jobs_runner_seer_iclr.sh) |
|---|---|---|
| 30313 | ctrl_vjepa_s0 | abl_m3soft_s0 |
| 31279 | ctrl_vjepa_s1 | abl_nobottleneck_s0 |
| 31866 | ctrl_vjepa_s2 | abl_m1active_s1 |
| 30674 GPU1 | ctrl_novjepa_s0 | ctrl_novjepa_s2 (chain) |
| 31511 | ctrl_novjepa_s1 | abl_nolic_s0 |
| 30594 | abl_randvjepa_s0 | abl_noretinex_s0, abl_randvjepa_s1 |
| 30555 | abl_m1active_s0 | abl_noigab_s0 |
No job is launched after 09-26 08:30. Started jobs are logged to seer_iclr_evals/jobs_started.txt.

## Data control
The indoor-only Seer (v35_S4_INDOOR_ONLY_ft) was initialised from the union Stage 2, so it is not indoor-only and is excluded. Retinexformer indoor-only and union were retrained from scratch on Seer's split (no leakage).

## Layout (23:30, approved by user)
Moved to appendix: training-set membership table (App. B), full temporal table (App. G, main keeps tLP+WE only via tab_temporal_compact), CP calibration table (App. F, tab_cp_archived; main text cites it). Main text currently ends p.7 with float-only pages because the prose is not written yet.

## Results snapshot 2026-09-26 11:00 (V-JEPA arm seeds 0,2; no-V-JEPA seeds 0,1)
- In-domain: pretrained V-JEPA vs zero tokens gives L0 +1.89 dB (sig), LPIPS -0.008 (sig); L1 +1.16 dB (n.s.), LPIPS -0.033 (sig); L0 tLP/WE both better (sig). Random-init V-JEPA is worse in-domain (L0 -2.43 dB vs pretrained).
- Cross-dataset: no gain. L2 PSNR -0.02, LPIPS +0.009 (worse, sig). L3-L5 LPIPS worse by 0.012-0.025. Random-init V-JEPA beats pretrained on L2 (18.46 vs 17.88).
- Ablations at 750 it (1 seed): opening the M1 gate hurts (-0.36 L0, -0.35 L1, -0.26 L2, all sig); w/o LIC L1 -1.27; w/o IGAB L1 +0.71; w/o temporal mixer L2 -0.30 and tLP worse; w/o Retinex aux ~0. CP-src is close to a no-op.
- Stratified: on L2, V-JEPA's PSNR gain correlates with input darkness (Spearman rho -0.61 vs LQ luma, p=0.013); L4 shows the same direction and L5 the opposite.
- => "Cross-Domain Prior" is NOT supported. Decision on title/framing needed from the user (asked 11:05).
- Launched s1 of nolic/noigab/nobottleneck (31511/30555/31866), done about 16:40. ctrl_vjepa_s1 eval re-run on 31279 (the first attempt failed on an SSL download).

## Final-pass checklist (after all runs, about 17:00): verify every directional claim in the prose
- 5.2: clean-6 order (Seer vs retrained RF; at 11:50 Seer 29.29 vs RF 29.07)
- 5.3: significance of the L0/L1/L2 deltas with 3 seeds; random-init better on L2?
- 5.4: ablation directions after the s1 seeds (nolic s1; the IGAB and bottleneck s1 runs were cancelled for B)
- 5.5: encoder ranking statements (archived runs, fixed)
- 5.6: temporal bests (tLP best on all three; WE best L0, within .001 on L1/L2)
- 5.7: CP-tgt Seer +0.85 on L2 vs baselines <= +0.34, which contradicts "similar size" -> rewrite
