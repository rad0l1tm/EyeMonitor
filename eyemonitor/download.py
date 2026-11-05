"""Explicit one-time download; inference itself never accesses the network."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request
import zipfile

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
             'face_landmarker/float16/1/face_landmarker.task')


def default_model():
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'eyemonitor/face_landmarker.task'


def main():
    p = argparse.ArgumentParser(description='Download official MediaPipe Face Landmarker v1')
    p.add_argument('--output', type=Path, default=default_model())
    p.add_argument('--sha256', help='Optional trusted SHA256 to verify the download')
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.download')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, tmp.open('wb') as f:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 50 * 1024 * 1024:
                    raise ValueError('Unexpected model size')
                f.write(chunk)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if args.sha256 and digest.lower() != args.sha256.lower():
            raise ValueError('SHA256 mismatch')
        if not zipfile.is_zipfile(tmp):
            raise ValueError('Model is not a MediaPipe task archive')
        tmp.replace(args.output)
        print(f'{args.output}\nSHA256 {digest}')
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
