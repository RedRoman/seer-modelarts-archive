"""Bounded official asset retrieval; authenticate bytes before any model loading."""
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path('/cache/zihaohe/seer_matched_predictor_assets_20261112_v1')
URL = 'http://dl.fbaipublicfiles.com/vjepa2/vitl.pt'
EXPECTED_SIZE = 5127726842
EXPECTED_SHA = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'
AUTHENTICATED_CHECKSUM_SOURCE = 'https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth'


def main():
    started = time.monotonic()
    received = 0
    partial = ROOT/'weights/vitl.pt.partial'
    final = ROOT/'weights/vitl.pt'
    report = {'download_url': URL, 'official_https_url': URL.replace('http:', 'https:'),
        'transport_note': 'HTTPS domain certificate expired on both tested machines; HTTP bytes must match the official HTTPS HF SHA before deserialization',
        'authenticated_checksum_source': AUTHENTICATED_CHECKSUM_SOURCE,
        'expected_sha256': EXPECTED_SHA, 'expected_bytes': EXPECTED_SIZE,
        'read_timeout_seconds': 15, 'overall_budget_seconds': 300,
        'dataset_pixels_read': 0, 'npu_used': False, 'model_loads': 0,
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        assert not final.exists() and not partial.exists(), 'Independent new file required'
        value = hashlib.sha256()
        last_progress = 0
        with urllib.request.urlopen(URL, timeout=15) as response, partial.open('xb') as output:
            assert response.status == 200
            assert int(response.headers['Content-Length']) == EXPECTED_SIZE
            report['headers'] = dict(response.headers)
            while True:
                assert time.monotonic()-started < 300, 'Bounded download time budget exceeded'
                block = response.read(1 << 20)
                if not block:
                    break
                output.write(block)
                value.update(block)
                received += len(block)
                assert received <= EXPECTED_SIZE
                if received-last_progress >= 128*(1 << 20):
                    print(json.dumps({'received': received, 'total': EXPECTED_SIZE,
                        'elapsed_seconds': time.monotonic()-started}), flush=True)
                    last_progress = received
        actual = value.hexdigest()
        report.update({'received_bytes': received, 'sha256': actual})
        assert received == EXPECTED_SIZE and actual == EXPECTED_SHA, 'Official asset identity mismatch; do not load'
        partial.rename(final)
        report.update({'status': 'authenticated', 'file': str(final)})
    except Exception as error:
        report.update({'status': 'failed_or_incomplete', 'received_bytes': received,
                       'error': type(error).__name__ + ': ' + str(error)})
    report['elapsed_seconds'] = time.monotonic()-started
    report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT/'evidence/download_receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
