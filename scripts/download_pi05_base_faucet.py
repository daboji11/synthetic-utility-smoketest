#!/usr/bin/env python3
"""Fetch official pi05_base params and attach frozen 54-H50 interface statistics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from pi05_base_faucet_config import (  # noqa: E402
    BASE, DATASET, expected_norm_stats, official_base_params_uri,
    verify_base_checkpoint, verify_source,
)


def main() -> None:
    from openpi.shared import download

    verify_source()
    uri = official_base_params_uri()
    stats = expected_norm_stats()
    if BASE.exists():
        print("PI05_BASE_READY " + json.dumps(verify_base_checkpoint()), flush=True)
        return
    # Official OpenPI URI is read from its pinned RoboCasa config. Only params are fetched.
    params = download.maybe_download(uri)
    if not (params / "_METADATA").is_file():
        raise FileNotFoundError("Official base params download is incomplete")
    BASE.mkdir(parents=True, exist_ok=False)
    (BASE / "params").symlink_to(params, target_is_directory=True)
    assets = BASE / "assets"
    assets.mkdir()
    norm = assets / "norm_stats.json"
    norm.write_text(json.dumps(stats, indent=2) + "\n")
    manifest = {
        "model": "pi05_base", "params_uri": uri,
        "params_cache": str(params.resolve()),
        "checkpoint_metadata_sha256": hashlib.sha256((params / "_METADATA").read_bytes()).hexdigest(),
        "interface_dataset": str(DATASET.resolve()),
        "interface_stats_source": str((DATASET / "meta/stats.json").resolve()),
        "norm_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
        "weights_untouched": True,
        "human300_weights_used": False,
    }
    (BASE / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_base_checkpoint()
    print("PI05_BASE_READY " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
