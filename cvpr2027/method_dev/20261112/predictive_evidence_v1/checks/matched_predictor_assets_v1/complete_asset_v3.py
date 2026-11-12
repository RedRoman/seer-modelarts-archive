"""One bounded final suffix request after the v2 total-transfer deadline; preserve partial evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path('/cache/zihaohe/seer_matched_predictor_assets_20261112_v1')
URL = 'http://dl.fbaipublicfiles.com/vjepa2/vitl.pt'
SIZE = 5127726842
SHA = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'


def main():
    began = time.monotonic()
    prefix = ROOT/'weights/vitl.pt.partial'
    prefix_size = prefix.stat().st_size
    chunk = (SIZE-prefix_size+7)//8
    segments = [prefix]
    resumed = []
    report = {'expected_sha256': SHA, 'expected_bytes': SIZE, 'read_timeout_seconds': 15,
              'tail_transfer_budget_seconds': 60, 'model_loads': 0, 'npu_used': False, 'dataset_pixels_read': 0}
    try:
        for index, start in enumerate(range(prefix_size, SIZE, chunk)):
            end = min(SIZE-1, start+chunk-1)
            path = ROOT/'weights/ranges_v2'/f'part_{index:02d}.bin'
            got = path.stat().st_size
            assert 0 < got <= end-start+1
            segments.append(path)
            if got != end-start+1:
                assert len(resumed) == 0, 'Only one incomplete suffix is authorized by this bounded recovery'
                tail_start = start+got
                tail = ROOT/'weights/ranges_v2'/f'part_{index:02d}.v3_tail'
                request = urllib.request.Request(URL, headers={'Range': f'bytes={tail_start}-{end}'})
                with urllib.request.urlopen(request, timeout=15) as response, tail.open('xb') as stream:
                    assert response.status == 206 and response.headers['Content-Range'] == f'bytes {tail_start}-{end}/{SIZE}'
                    while True:
                        assert time.monotonic()-began < 60, 'Final suffix transfer budget exceeded'
                        block = response.read(1 << 20)
                        if not block:
                            break
                        stream.write(block)
                assert tail.stat().st_size == end-tail_start+1
                resumed.append({'index': index, 'start': tail_start, 'end': end, 'bytes': tail.stat().st_size})
                segments.append(tail)
        joined = ROOT/'weights/vitl.pt.assembled_v3'
        value = hashlib.sha256()
        with joined.open('xb') as output:
            for path in segments:
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        output.write(block)
                        value.update(block)
        report.update({'sha256': value.hexdigest(), 'bytes': joined.stat().st_size, 'resumed_ranges': resumed})
        assert report['sha256'] == SHA and report['bytes'] == SIZE, 'Official HTTPS checksum identity mismatch; do not load'
        final = ROOT/'weights/vitl.pt'
        assert not final.exists()
        joined.rename(final)
        report.update({'status': 'authenticated', 'file': str(final)})
    except Exception as error:
        report.update({'status': 'failed_or_incomplete', 'error': type(error).__name__+': '+str(error)})
    report['elapsed_seconds'] = time.monotonic()-began
    report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT/'evidence/download_v3_receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
