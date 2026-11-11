DID public acquisition metadata — 2026-11-11

The author's public Baidu share code `nomy` is accepted anonymously. No account login or CAPTCHA was required for the completed file listing. The share exposes one DID directory with exactly two ZIP files and no single-video directories or multipart volumes:

| File | Reported bytes | Decimal GB | GiB |
| --- | ---: | ---: | ---: |
| DID_1080.zip | 32,795,182,798 | 32.795 | 30.543 |
| DID_2k.zip | 56,650,437,780 | 56.650 | 52.760 |
| Both | 89,445,620,578 | 89.446 | 83.303 |

The complete successful inventory is `baidu_file_inventory.json` (SHA256 212826212e3ab4823b4a9c762fc806b9935a630339d68a546c06c466bdc13e1a). The earlier probes are retained as partial attempts. The page initially labels a virtual share path; the normal root-list API supplies the actual directory path used for the successful subdirectory listing. Numeric metadata fields were normalized from the server's string representation. No session-key, cookie value or download URL was persisted.

The public share does not expose a test87/validation archive or individual video selection. Either resolution ZIP can be selected independently, so obtaining both 89 GB is unnecessary. Which video IDs, internal directories and frame formats each ZIP contains remains unverified: this metadata phase did not request download links, file bytes or ZIP central directories. Filenames alone do not prove the two ZIPs contain identical videos or establish their resolutions. The server's `md5` strings are opaque metadata, not independently verified file checksums.

OpenXLab/OpenDataLab still exposes only directory metadata for the official ciki/DID entry. Anonymous v2/v3 info requests return id8134, public state, fileSize89,445,620,691 bytes and fileNum0. Anonymous v2/v3 file lists return HTTP200 but `hasMediaFile:false`, `total:0`, `list:null`, with an explicit direct-download-unavailable notice. Its `downloadAuth` endpoint returns `noAuthRequired`. Therefore lack of account authorization is not the observed mirror blocker. The catalog size is a planning estimate, not a separately verified downloadable file size.

The official GitHub train and test lists were fetched anonymously: 238 unique train IDs and 87 unique test IDs. SHA256 values are e2053f68bb0d30475bfa0df0cc78bb57c347a99e4ffcdec9802c46d0fd73c6e4 and 25fcaec14713d806b9cbaf6e0e2ecea7699bb7122b622873fe7875c92019cf17. No official validation list is present. The AMNet 330/83 reference split differs and must not be substituted for the official split.

Primary references: [official DID repository](https://github.com/ciki000/DID), [official split directory](https://github.com/ciki000/DID/tree/main/DID_dataset), [author-provided OpenDataLab alternative](https://github.com/ciki000/DID/issues/2#issuecomment-1858787035), [author-published Baidu share](https://pan.baidu.com/s/1dSx-XxKd7MVwh9xPLvZn3A?pwd=nomy). The official README and issues checked expose no additional author-provided mirror.

`anonymous_http_v2.json`, `anonymous_http_v3_baidu.json` and `anonymous_permission_baidu_init.json` preserve the public catalog responses. `sdk_source_notes.json` records read-only inspection of the 314,531-byte official openxlab0.1.3 wheel in memory: no SDK install or execution. The SDK supports anonymous info/list but stops on `hasMediaFile:false`. `official_split_metadata.json` records split-file requests. Public share forms were subsequently submitted under explicit authorization; only sanitized responses and file metadata were saved. No user account credentials were read, no CAPTCHA was solved, and no video/archive or download link was requested during this metadata phase.

The subsequent authorized ordinary-download eligibility attempt on host32193 is archived separately in `../did_download_v1/`. Fresh anonymous verification and both directory-list requests returned HTTP200/errno0, but the public file page provided no ordinary-download sign/timestamp. The helper stopped before sending `/api/sharedownload`; no URL or video bytes were requested, and no ZIP/partial ZIP was created. This is a missing anonymous download-signature observation, not evidence that login is necessarily required. No further interface was enumerated and no acquisition remains running.
