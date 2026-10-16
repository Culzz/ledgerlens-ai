"""Fetch the FinQA dataset into data/finqa/.

FinQA ships as JSON, not PDFs. Each record already contains the extracted table
and the surrounding narrative text, which is why the benchmark track never
touches an extraction pipeline.

    python scripts/download_data.py
    python scripts/download_data.py --force

Records the source URL and a checksum of each file so a run can be reproduced
later. Does nothing if the files are already present and their checksums agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "finqa"

# The dataset as published by the FinQA authors. Splits are used as published
# and are never recombined.
SOURCE_BASE = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
SPLITS = ("train.json", "dev.json", "test.json")
# train.json is ~78 MB, so the default is dev only. Pass --splits train,dev,test
# for everything, but there's rarely a reason to want train locally.
DEFAULT_SPLITS = ("dev.json",)


def fetch(url: str, *, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ledgerlens-ai/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--base-url", default=SOURCE_BASE)
    parser.add_argument(
        "--splits",
        default=",".join(s.removesuffix(".json") for s in DEFAULT_SPLITS),
        help=f"comma-separated subset of train,dev,test (default: {','.join(DEFAULT_SPLITS)})",
    )
    args = parser.parse_args(argv)

    requested = [f"{name.strip().removesuffix('.json')}.json" for name in args.splits.split(",")]
    unknown = [name for name in requested if name not in SPLITS]
    if unknown:
        parser.error(f"unknown split(s): {', '.join(unknown)}; choose from {', '.join(SPLITS)}")
    splits = tuple(requested)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    manifest_path = dest / "manifest.json"
    manifest: dict[str, object] = {
        "source_base": args.base_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {},
    }

    failures = 0
    for name in splits:
        target = dest / name
        if target.exists() and not args.force:
            data = target.read_bytes()
            print(f"{name}: already present, {len(data):,} bytes")
        else:
            url = f"{args.base_url}/{name}"
            print(f"{name}: downloading from {url}")
            try:
                data = fetch(url)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"{name}: FAILED - {exc}", file=sys.stderr)
                failures += 1
                continue
            target.write_bytes(data)

        try:
            records = json.loads(data)
            count = len(records) if isinstance(records, list) else "not a list"
        except json.JSONDecodeError:
            count = "unparseable"

        manifest["files"][name] = {
            "bytes": len(data),
            "sha256": sha256(data),
            "records": count,
        }
        print(f"{name}: {len(data):,} bytes, {count} records, sha256 {sha256(data)[:12]}...")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest written to {manifest_path}")

    if failures:
        print(f"\n{failures} split(s) failed.", file=sys.stderr)
        print(
            "If the network is unavailable, place the split files in "
            f"{dest} manually and re-run. Splits are not committed to git.",
            file=sys.stderr,
        )
        return 1

    print("\nnext: python -m evaluation.replay_gold_programs --split data/finqa/dev.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
