#!/usr/bin/env python3
"""Export the source context64 character scorer as a fixed B9 FP32 OpenVINO IR.

Install scripts/accelerator-requirements.txt in a private virtual environment.
The new output directory binds the source weights, config, vocabulary and license
to both IR files; the source artifact is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import openvino as ov
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/chinese-ime-training/neural"))
from initialize import load_initial_model


class Scorer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, target_mask):
        hidden = self.model.tok(tokens) + self.model.pos(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.model.blocks:
            hidden = block(hidden)
        hidden = self.model.ln_f(hidden)
        logits = self.model.head(hidden)
        targets = torch.cat((tokens[:, 1:], tokens[:, :1]), dim=1)
        # The tied embedding avoids gathering a B*W*vocab intermediate on NPU.
        selected = (hidden * self.model.tok(targets)).sum(-1)
        # Local maxima prevent small FP16 exponentials in a full-vocab reduction.
        grouped = torch.nn.functional.pad(logits, (0, (-logits.shape[-1]) % 256), value=-float("inf"))
        grouped = grouped.reshape(tokens.shape[0], tokens.shape[1], -1, 256)
        probabilities = selected - grouped.logsumexp(-1).logsumexp(-1)
        mask = torch.cat((target_mask, torch.zeros_like(target_mask[:, :1])), dim=1)
        return (probabilities * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory; existing destinations are rejected")
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    source_sha = digest(args.source)
    torch.set_num_threads(4)
    model, cfg, vocab, metadata = load_initial_model(args.source)
    if cfg.context != 64:
        raise ValueError("the accelerator scorer requires source context64")
    scorer = Scorer(model.eval()).eval()
    inputs = (torch.full((9, 64), 2, dtype=torch.int64), torch.ones((9, 63), dtype=torch.float32))
    with torch.no_grad():
        ir = ov.convert_model(scorer, example_input=inputs)
    for port, name in zip(ir.inputs, ["tokens", "target_mask"]):
        port.get_tensor().set_names({name})
    ir.output(0).get_tensor().set_names({"scores"})
    ir.reshape({"tokens": [9, 64], "target_mask": [9, 63]})
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        xml = staging / args.source.with_suffix(".xml").name
        ov.save_model(ir, xml, compress_to_fp16=False)
        if digest(args.source) != source_sha:
            raise ValueError("source artifact changed during export")
        manifest = {
            "format": "yanliu-ranker-ir-v1", "batch": 9,
            "source_sha256": source_sha, "xml_sha256": digest(xml),
            "bin_sha256": digest(xml.with_suffix(".bin")),
            "config": json.loads(metadata["config"]), "vocab": vocab,
            "license": metadata["license"], "attribution": metadata["attribution"],
            "openvino_version": ov.__version__,
        }
        xml.with_suffix(".openvino.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        staging.rename(args.output_dir)
    print(json.dumps({key: value for key, value in manifest.items() if key != "vocab"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
