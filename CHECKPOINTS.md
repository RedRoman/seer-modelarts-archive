# Checkpoints and recovery

## CVPR final models (backup verified 2026-11-11)

The 25 reachable final models from E17/E18/E20/E26 are backed up with their
logs, evaluation files and qualitative assets. For each run, `final_cpu.pt`
is preferred over `final.pt` when available. Datasets and the public V-JEPA
weights are excluded. This is a recovery archive, not a new public Release.

- Archive: `/temp/zihaohe/seer_cvpr/recovery_20261111/seer_converged_31051_20261111.tar`
- Reachable through hosts mounting the shared SFS, including 31051.
- Size: 2,257,909,760 bytes; 135 files, 25 checkpoints.
- SHA-256: `8397ef128529888a5e151c0ba22dbb8b58d17046ad04c469777025f523f8dfa6`
- Per-file manifest: [checkpoint_manifest_31051.json](cvpr2027/results/protocol_audit_20261111/checkpoint_manifest_31051.json).
- Reproducible backup tool: [archive_converged.py](cvpr2027/tools/archive_converged.py).

Verify the archive before extracting it into a new directory. Then verify the
extracted members against `artifact_manifest.json`. The archive contains only
available hosts' weights; it does not recover E19/E21/E22/E24/E25/E2c/E28
weights from the unreachable 32677 local disk.

The attempted full local download did not complete at the available transfer
rate. A partial local file is not a valid backup; the independently verified
shared SFS archive and the committed hash manifest are the completed copies.

## ModelArts snapshot (2026-09-27)

The historical [ckpt-seer Release](https://github.com/RedRoman/seer-modelarts-archive/releases/tag/ckpt-seer)
contains September model and ablation weights split into numbered parts.
These are not the later CVPR final models. GitHub exposes a SHA-256 digest for
each uploaded asset in the Release API; verify parts before concatenating
them in numbered order. Do not use a September weight as a substitute for a
missing CVPR run merely because its architecture or name is similar.
