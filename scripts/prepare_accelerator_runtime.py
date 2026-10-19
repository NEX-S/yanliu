#!/usr/bin/env python3
"""Prepare the pinned official OpenVINO GPU/NPU runtime in a new directory.

The default downloads and verifies the official archive. --archive reuses a local
copy only after checking the same SHA256. No system packages or services change.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.request

URL = "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.0/linux/openvino_genai_ubuntu24_2026.0.0.0_x86_64.tar.gz"
SHA256 = "d4804ac4b0107806146a44f5fca5622c69da6001a6b4f09174079bfac1450dbc"
LIBRARY = re.compile(r"lib(?:openvino(?:_c|_ir_frontend)?\.so(?:\.[\d.]+)?|openvino_intel_(?:gpu_plugin|npu_plugin|npu_compiler)\.so|tbb(?:bind_2_5|malloc(?:_proxy)?)?\.so(?:\.[\d.]+)?|hwloc\.so(?:\.[\d.]+)?)")
REQUIRED = {"libopenvino.so.2600", "libopenvino_c.so", "libopenvino_ir_frontend.so.2600", "libtbb.so.12", "libhwloc.so.15", "libopenvino_intel_gpu_plugin.so", "libopenvino_intel_npu_plugin.so", "libopenvino_intel_npu_compiler.so", "licenses/LICENSE", "licenses/runtime-third-party-programs.txt", "licenses/onetbb_third-party-programs.txt"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new directory; existing destinations are rejected")
    parser.add_argument("--archive", type=Path, help="already downloaded official archive; SHA256 is still verified")
    args = parser.parse_args()
    destination = args.destination
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"runtime directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = args.archive
        if archive is None:
            archive = temporary / "runtime.tar.gz"
            request = urllib.request.Request(URL, headers={"User-Agent": "yanliu-assets/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        if digest(archive) != SHA256:
            raise ValueError("official OpenVINO archive SHA256 mismatch")
        staging = temporary / "runtime"
        staging.mkdir()
        selected = {}
        with tarfile.open(archive) as package:
            for member in package.getmembers():
                path = PurePosixPath(member.name)
                name = path.name
                if LIBRARY.fullmatch(name):
                    relative = name
                elif "licensing" in path.parts:
                    relative = str(PurePosixPath("licenses", *path.parts[path.parts.index("licensing") + 1:]))
                    if "tokenizers" in path.parts or "genai" in name.lower():
                        continue
                else:
                    continue
                if member.isdir():
                    continue
                if relative in selected or ".." in PurePosixPath(relative).parts:
                    raise ValueError(f"duplicate or invalid archive output path: {relative}")
                selected[relative] = member
            missing = REQUIRED - selected.keys()
            if missing:
                raise ValueError(f"official archive is missing runtime files: {sorted(missing)}")
            for relative, member in selected.items():
                output = staging / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                if member.issym():
                    target = PurePosixPath(member.linkname)
                    if len(target.parts) != 1 or str(target) not in selected:
                        raise ValueError(f"invalid runtime library alias: {member.name} -> {member.linkname}")
                    output.symlink_to(str(target))
                elif member.isfile():
                    with package.extractfile(member) as source, output.open("wb") as stream:
                        shutil.copyfileobj(source, stream)
                else:
                    raise ValueError(f"unsupported runtime archive entry: {member.name}")
        manifest = {"format": "yanliu-runtime-v1", "abi": "official-archive-ABI1", "source": {"url": URL, "sha256": SHA256}, "files": {name: digest(staging / name) for name in sorted(selected)}}
        (staging / "runtime-source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"runtime directory already exists: {destination}")
        staging.rename(destination)
    print(json.dumps({"destination": str(destination), "source": manifest["source"], "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
