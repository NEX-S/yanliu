#!/usr/bin/env python3
"""Repeat the fixed public 25M context64 training and mixed-dev selection."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def derive_initial(source, output, expected_sha):
    if digest(source) != expected_sha:
        raise ValueError("upstream model SHA-256 does not match recipe.json")
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(source, framework="pt", device="cpu") as model:
        metadata = model.metadata()
        config = json.loads(metadata["config"])
        if config["context"] != 128:
            raise ValueError("expected the pinned upstream context128 model")
        config["context"] = 64
        tensors = {name: model.get_tensor(name) for name in model.keys()}
        tensors["pos.weight"] = tensors["pos.weight"][:64].contiguous()
    metadata.update(
        config=json.dumps(config, separators=(",", ":")),
        parent_model_sha256=expected_sha,
        window_derivation="context128 to context64; pos.weight first64 rows; all other tensor bytes unchanged")
    save_file(tensors, str(output), metadata)


def select_checkpoint(cases, parent_report, reports, helper):
    if reports[0]["rows"] != parent_report["rows"]:
        raise ValueError("epoch0 roundtrip changed mixed development rows")
    comparisons = [helper.compare_reports(cases, parent_report, report) for report in reports]
    # range order makes the first maximum the earliest epoch, including epoch0.
    epoch = max(range(4), key=lambda i: Fraction(*comparisons[i]["group_top1_fraction"]))
    return epoch, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/yanliu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, binary, output = (args.upstream_model.resolve(), args.binary.resolve(),
                              args.output_dir.resolve())
    recipe = json.loads((HERE / "recipe.json").read_text())
    for name, record in recipe["data"].items():
        if digest(HERE / name) != record["sha256"]:
            raise ValueError(f"frozen data SHA-256 does not match: {name}")
    for name, expected in recipe["historical_training"]["dictionaries"].items():
        if digest(ROOT / "data/dictionaries" / name) != expected:
            raise ValueError(f"dictionary SHA-256 does not match: {name}")
    output.mkdir(parents=True, exist_ok=False)
    initial = output / "initial-context64.safetensors"
    derive_initial(source, initial, recipe["upstream"]["sha256"])
    training = output / "training"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/train_ranker.py"),
        "--model", str(initial), "--binary", str(binary),
        "--train", str(HERE / "train.jsonl"), "--dev", str(HERE / "novel-dev.jsonl"),
        "--output-dir", str(training), "--epochs", "3", "--lr", "0.00001",
        "--seed", "20261019", "--batch", "4", "--threads", "2"], cwd=ROOT, check=True)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_ranker as helper
    cases = helper.read_cases(HERE / "mixed-dev.jsonl")
    models = [initial] + [training / f"epoch{epoch:02d}-int8.safetensors" for epoch in range(4)]
    reports = []
    for name, model in zip(["parent", "epoch00", "epoch01", "epoch02", "epoch03"], models, strict=True):
        report = output / f"mixed-{name}.json"
        subprocess.run([
            str(binary), "--dictionary-dir", str(ROOT / "data/dictionaries"),
            "--model", str(model), "--device", "cpu", "evaluate",
            "--data", str(HERE / "mixed-dev.jsonl"), "--output", str(report)], cwd=ROOT, check=True)
        reports.append(json.loads(report.read_text()))
    epoch, comparisons = select_checkpoint(cases, reports[0], reports[1:], helper)
    selected = initial if epoch == 0 else models[epoch + 1]
    shutil.copyfile(selected, output / "selected.safetensors")
    selection = {
        "epoch": epoch,
        "model_sha256": digest(selected),
        "training_cli_epoch": json.loads((training / "selection.json").read_text())["epoch"],
        "mixed_dev": [{"epoch": i, "group_top1_fraction": row["group_top1_fraction"],
                       "metrics": row["metrics"]} for i, row in enumerate(comparisons)],
        "scope": "Observed development selection; fresh recipe metadata and output bytes may differ from the release."}
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection))


if __name__ == "__main__":
    main()
