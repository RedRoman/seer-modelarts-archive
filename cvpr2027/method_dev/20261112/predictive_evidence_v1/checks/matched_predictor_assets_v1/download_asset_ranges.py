"""Resume our stopped prefix via bounded HTTP ranges; authenticate against official HF SHA."""
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    parts_root = ROOT/'weights/ranges_v2'
    parts_root.mkdir(exist_ok=False)
    report = {'url': URL, 'expected_sha256': SHA, 'size': SIZE, 'prefix_bytes': prefix_size,
        'threads': 8, 'overall_transfer_budget_seconds': 180, 'socket_timeout_seconds': 15,
        'authenticated_checksum_source': 'https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/raw/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth',
        'dataset_pixels_read': 0, 'npu_used': False, 'model_loads': 0}
    chunk = (SIZE-prefix_size+7)//8
    intervals = [(start, min(SIZE-1, start+chunk-1)) for start in range(prefix_size, SIZE, chunk)]

    def fetch(index, start, end):
        request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
        path = parts_root/f'part_{index:02d}.bin'
        got = 0
        with urllib.request.urlopen(request, timeout=15) as response, path.open('xb') as stream:
            assert response.status == 206
            assert response.headers['Content-Range'] == f'bytes {start}-{end}/{SIZE}'
            while True:
                assert time.monotonic()-began < 180, 'Bounded range transfer budget exceeded'
                block = response.read(1 << 20)
                if not block:
                    break
                stream.write(block)
                got += len(block)
        assert got == end-start+1
        return {'index': index, 'start': start, 'end': end, 'bytes': got, 'file': str(path)}

    try:
        parts = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, index, *interval) for index, interval in enumerate(intervals)]
            for future in as_completed(futures):
                part = future.result()
                parts.append(part)
                print(json.dumps({'part_complete': part['index'], 'bytes': part['bytes'],
                                  'elapsed_seconds': time.monotonic()-began}), flush=True)
        joined = ROOT/'weights/vitl.pt.assembled'
        value = hashlib.sha256()
        with joined.open('xb') as output:
            for path in [prefix]+[Path(row['file']) for row in sorted(parts, key=lambda row: row['index'])]:
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        output.write(block)
                        value.update(block)
        report.update({'sha256': value.hexdigest(), 'bytes': joined.stat().st_size, 'parts': parts})
        assert report['sha256'] == SHA and report['bytes'] == SIZE, 'Identity mismatch: do not load'
        final = ROOT/'weights/vitl.pt'
        assert not final.exists()
        joined.rename(final)
        report.update({'status': 'authenticated', 'file': str(final)})
    except Exception as error:
        report.update({'status': 'failed_or_incomplete', 'error': type(error).__name__+': '+str(error)})
    report['elapsed_seconds'] = time.monotonic()-began
    report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT/'evidence/download_v2_receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
