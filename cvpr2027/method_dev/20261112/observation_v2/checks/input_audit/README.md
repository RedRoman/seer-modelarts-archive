BVI VAL16 target input post-verification

The frozen auditor verified the same 512 selected original LQ+GT files and actual center256 RGB uint8 crops from 2026-11-12T02:19:34.063291+00:00 to 2026-11-12T02:20:24.252944+00:00. It completed with exit0, content_unchanged=true and an empty changes list. The original files total1,171,529,371 bytes; the two full content inventories are exactly equal, including all32 THWC clip tensors. Independent local verification rehashed both downloaded manifests, checked their sidecars/canonical content hashes, coverage, byte sums, tensor shapes and exact full-content equality. No images were copied into the evidence archive.

Post manifest SHA256: 21d6fc48f656d3ba72a6c4c735b30d2de23b5c60d419afabf8866b6b371c4967
Shared content SHA256: f0666763194c845d41297bfd9e5c22fdcda75971f7a1cc106f85cec559d6591d
Frozen auditor SHA256: b85edbceef916d11ad51b043ca472bee7e20052700a2e54f3341e45a6e4e60ed

Capture timestamps are explicit: 2026-11-12T02:04:35.951795+00:00 to 2026-11-12T02:05:26.828059+00:00. Initial capture overlapped early main evaluation and followed the Temp reference, so this is endpoint content stability evidence and does not retrospectively hash every earlier inference instant. The post check began only after runtime confirmed all three evaluations finished. No Torch, model, scoring, NPU or TEST-image access occurred in this audit.
