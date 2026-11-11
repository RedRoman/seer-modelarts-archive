DID_1080 ordinary anonymous acquisition attempt — 2026-11-11

Outcome: stopped before requesting a download URL. The fresh anonymous public share page did not supply `sign` and `timestamp`, required by the ordinary browser `/api/sharedownload` call. No sharedownload request was sent. This observation does not establish whether account login is required: no login or CAPTCHA challenge occurred during the completed requests, and no further interface was tried.

On host32193, share initialization, author-code verification, root listing and DID directory listing all returned HTTP200; all three JSON API results had errno0. The target identity was confirmed as DID_1080.zip, 32,795,182,798 bytes. Available disk space at start was 4,008,355,524,608 bytes. The process exited2 at 2026-11-11T15:47:28Z with `normal_download_signature_not_available`. It created neither the ZIP nor a partial ZIP, transferred no video bytes, and performed no extraction or test-video access.

Only the ordinary anonymous public flow was used with a new in-memory cookie jar. No user credential was read, no account login was attempted, no CAPTCHA was solved, no native-client impersonation was used, and no additional download/signature endpoint was enumerated. Session tokens, cookies and returned URLs are absent from the evidence. The companion script records only whitelisted statuses and scalar metadata, and stops rather than logging exception text.

Evidence: `status.json` and `progress.jsonl` were copied unchanged from `/cache/zihaohe/seer_method_dev_20261111/datasets/DID_download_v1/`. The reviewed standard-library helper is `acquire_did_1080.py`, SHA256 43fbc6847eaa44347db9202f636984a2bbbe272e3d8379bb11eb78e49ea87657, also retained remotely under sibling `DID_download_tool_v1/`. The separate completed public file inventory is in `../did_access_metadata_20261111/baidu_file_inventory.json`.

The share provides only DID_1080.zip and DID_2k.zip, with no exposed test87-only ZIP, per-video files or multipart volumes. Archive contents remain unverified. This attempt is complete; no background acquisition remains running.
