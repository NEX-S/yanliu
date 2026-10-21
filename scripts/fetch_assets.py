#!/usr/bin/env python3
"""Fetch or verify every asset pinned in data/manifest.json; stdlib only."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def apply_patch(original, patch):
    """Apply a unified diff to the pinned source; the caller verifies the effective hash."""
    source = original.decode("utf-8").splitlines(keepends=True)
    output, position = [], 0
    for line in patch.decode("utf-8").splitlines(keepends=True)[2:]:
        if line.startswith("@@"):
            start = int(re.match(r"@@ -(\d+)", line)[1]) - 1
            output += source[position:start]
            position = start
        elif line.startswith("+"):
            output.append(line[1:])
        else:
            if line.startswith(" "):
                output.append(line[1:])
            position += 1
    return "".join(output + source[position:]).encode("utf-8")


def fetch(asset, verify_only):
    path = ROOT / asset["path"]
    expected = asset.get("effective_sha256", asset["sha256"])
    expected_size = asset.get("effective_size", asset["size"])
    patch = None
    if "local_patch" in asset:
        patch = (ROOT / asset["local_patch"]["path"]).read_bytes()
        if hashlib.sha256(patch).hexdigest() != asset["local_patch"]["sha256"]:
            raise ValueError(f"local patch checksum mismatch: {asset['local_patch']['path']}")
    if path.is_file() and path.stat().st_size == expected_size and digest(path) == expected:
        print(f"verified {asset['path']}")
        return
    if verify_only:
        raise ValueError(f"missing or mismatched asset: {asset['path']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Temporary file lives beside its destination, so publication is atomic.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(asset["url"], headers={
                "User-Agent": "yanliu-assets/0.1", "Accept": "application/octet-stream"})
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, output)
        if temporary.stat().st_size != asset["size"] or digest(temporary) != asset["sha256"]:
            raise ValueError(f"download checksum/size mismatch: {asset['path']}")
        if patch is not None:
            temporary.write_bytes(apply_patch(temporary.read_bytes(), patch))
        if temporary.stat().st_size != expected_size or digest(temporary) != expected:
            raise ValueError(f"patched asset checksum/size mismatch: {asset['path']}")
        temporary.replace(path)
        print(f"downloaded {asset['path']}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="verify local files without downloading")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported asset manifest schema")
    for asset in manifest["assets"]:
        fetch(asset, args.verify_only)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"asset error: {error}", file=sys.stderr)
        sys.exit(1)
