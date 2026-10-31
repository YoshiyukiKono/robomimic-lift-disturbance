"""Download one identified upstream checkpoint and verify it before loading."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

MODEL_URL = "https://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_low_dim_epoch_1000_succ_100.pth"
MODEL_SHA256 = "3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605"
MODEL_BYTES = 7959022
DEFAULT_PATH = Path("models/lift_ph_low_dim.pth")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run python -m lift_tutorial.checkpoint")
    digest = sha256_file(path)
    if digest != MODEL_SHA256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {digest}. Expected {MODEL_SHA256}. File was not loaded.")
    return {"url": MODEL_URL, "sha256": digest, "bytes": path.stat().st_size,
            "name": "robomimic v0.1 Lift PH low_dim BC-RNN"}


def download(path=DEFAULT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        metadata = verify_checkpoint(path)
        print(f"Verified existing checkpoint: {path}")
        return metadata
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "robomimic-lift-tutorial/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                count = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > MODEL_BYTES:
                        raise ValueError("Unexpected checkpoint size; download stopped")
                    stream.write(block)
        metadata = verify_checkpoint(temporary)
        os.replace(temporary, path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Downloaded and SHA256 verified: {path} ({MODEL_BYTES:,} bytes)")
        return metadata
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
