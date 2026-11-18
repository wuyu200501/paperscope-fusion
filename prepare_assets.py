"""Fetch the exact dataset archive and model revisions used in PaperScope."""

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARCHIVE = DATA / "qasper-train-dev-v0.3.tgz"
QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
MODELS = [
    ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", [
        "config.json", "config_sentence_transformers.json", "model.safetensors",
        "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json",
    ]),
    ("deepset/minilm-uncased-squad2", "934656cdda79824eabf503ed56e15c01ddbdbe3f", [
        "config.json", "model.safetensors", "special_tokens_map.json",
        "tokenizer_config.json", "vocab.txt",
    ]),
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_archive():
    if ARCHIVE.exists() and digest(ARCHIVE) == QASPER_SHA256:
        return
    partial = ARCHIVE.with_suffix(f".{os.getpid()}.tmp")
    try:
        with urlopen(QASPER_URL, timeout=90) as source, partial.open("wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        if digest(partial) != QASPER_SHA256:
            raise ValueError("QASPER archive checksum mismatch.")
        partial.replace(ARCHIVE)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check local assets without networking.")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")
    if not args.check:
        from huggingface_hub import snapshot_download

        prepare_archive()
        for repo, revision, files in MODELS:
            name = repo.split("/")[-1]
            print(f"Preparing {name} at {revision[:8]}", flush=True)
            snapshot_download(repo, revision=revision, local_dir=DATA / name,
                              allow_patterns=files, max_workers=3)
    if not ARCHIVE.exists() or digest(ARCHIVE) != QASPER_SHA256:
        raise SystemExit("QASPER is missing or has a different checksum.")
    for repo, _, files in MODELS:
        name = repo.split("/")[-1]
        for file in files:
            asset = DATA / name / file
            if not asset.is_file() or not asset.stat().st_size:
                raise SystemExit(f"Missing asset {name}/{file}. Run prepare_assets.py.")
    print("Dataset checksum and model files verified.")


if __name__ == "__main__":
    main()
