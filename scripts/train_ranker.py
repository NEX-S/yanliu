#!/usr/bin/env python3
"""Fine-tune the current character model on real CLI candidate shortlists.

Install scripts/training-requirements.txt in a private virtual environment.
Only explicitly supplied train/dev JSONL and optional text files are read. The fresh output
directory retains every development case and all epoch artifacts; this command
does not install a model or access system learning.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DICTIONARIES = ROOT / "data/dictionaries"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_cases(path):
    cases, ids = [], set()
    for number, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            case = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        if not isinstance(case, dict):
            raise ValueError(f"{path}:{number}: case must be a JSON object")
        if "gold" in case and "golds" in case:
            raise ValueError(f"{path}:{number}: supply either gold or golds, not both")
        golds = case.get("gold", case.get("golds"))
        if isinstance(golds, str):
            golds = [golds]
        if (not isinstance(case.get("input"), str) or not case["input"]
                or not isinstance(case.get("context", ""), str)
                or not isinstance(golds, list) or not golds
                or any(not isinstance(gold, str) or not gold for gold in golds)):
            raise ValueError(f"{path}:{number}: nonempty input/gold and string context required")
        identity = case.get("id", "")
        if not isinstance(identity, str):
            raise ValueError(f"{path}:{number}: case id must be a string")
        identity = identity or str(number)
        if identity in ids:
            raise ValueError(f"{path}:{number}: case id must be a unique string")
        if "group_id" in case and (not isinstance(case["group_id"], str) or not case["group_id"]):
            raise ValueError(f"{path}:{number}: group_id must be a nonempty string")
        if "source_group" in case and (not isinstance(case["source_group"], str) or not case["source_group"]):
            raise ValueError(f"{path}:{number}: source_group must be a nonempty string")
        ids.add(identity)
        cases.append(dict(case, id=identity, golds=golds, context=case.get("context", "")))
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def reject_overlap(train, dev):
    def normalized(text):
        return "".join(char for char in unicodedata.normalize("NFKC", text).casefold()
                       if char.isalnum())

    def identities(cases):
        return {(normalized(case["input"]), normalized(case.get("context", "")), normalized(gold))
                for case in cases for gold in case["golds"]}

    groups = {case["source_group"] for case in train if "source_group" in case} & {
        case["source_group"] for case in dev if "source_group" in case}
    duplicates = identities(train) & identities(dev)
    if groups or duplicates:
        raise ValueError(f"train/dev overlap: {len(groups)} source_group(s), {len(duplicates)} duplicate(s)")


def shortlist(case, candidates):
    """Match src/model.rs with personal learning disabled, as in CLI evaluate."""
    def answers(candidate):
        return (candidate["consumed"] == len(case["input"].encode("utf-8"))
                and candidate["source"] != "Completion")

    if not candidates or not answers(candidates[0]):
        return []
    leader = candidates[0]["source"]
    if ((len(case["input"]) == 1 and case["input"].lower() in "bpmfdtnlgkhjqxrzcsyw")
            or leader == "User" or (not case["context"] and leader == "ExactDictionary")):
        return []
    eligible = [i for i, candidate in enumerate(candidates) if answers(candidate)]
    selected = eligible[:7]
    for index in eligible[7:]:
        if len(selected) == 9:
            break
        if len(candidates[index]["text"]) not in {
                len(candidates[chosen]["text"]) for chosen in selected}:
            selected.append(index)
    for index in eligible[7:]:
        if len(selected) == 9:
            break
        if index not in selected:
            selected.append(index)
    return selected


def static_score(candidate):
    score = struct.unpack("<f", struct.pack("<I", candidate["score_bits"]))[0]
    if not math.isfinite(score):
        raise ValueError("nonfinite static candidate score")
    return score


def aligned_rows(cases, report):
    if len(cases) != len(report["rows"]):
        raise ValueError("CLI report changed the case denominator")
    for case, row in zip(cases, report["rows"], strict=True):
        if case["id"] != row["id"]:
            raise ValueError(f"CLI case order/id mismatch: {case['id']} != {row['id']}")
    return report["rows"]


def training_groups(cases, report, window, vocab):
    groups, statuses = [], Counter()
    characters = set(vocab[3:])
    for case, row in zip(cases, aligned_rows(cases, report), strict=True):
        candidates = row["candidates"]
        indices = shortlist(case, candidates)
        texts = [candidates[index]["text"] for index in indices]
        known = [all(char in characters for char in text) for text in texts]
        if len(indices) < 2 or len(set(texts)) < 2:
            status = "protected_or_no_competition"
        elif any(len(text) >= window or any(ch.isascii() and ch.isalpha() for ch in text)
                 for text in texts):
            status = "neutral_language_group"
        elif not known[0]:
            status = "neutral_language_group_oov_leader"
        elif sum(known) < 2:
            status = "neutral_language_group_insufficient_known_candidates"
        elif not any(text in case["golds"] for text in texts):
            status = "gold_outside_shortlist"
        elif not any(is_known and text in case["golds"]
                     for text, is_known in zip(texts, known, strict=True)):
            status = "gold_only_unrepresented"
        elif all(text in case["golds"] for text in texts):
            status = "no_negative_candidate"
        else:
            status = "trainable"
            prior = candidates[0]["source"] in {"ExactDictionary", "User"}
            leader = static_score(candidates[0])
            penalties = [2.0 * max(leader - static_score(candidates[index]), 0.0)
                         if prior else 0.0 for index in indices]
            groups.append({"id": case["id"], "context": case["context"], "texts": texts,
                           "static_indices": indices, "known_mask": known, "penalties": penalties,
                           "gold_indices": [i for i, text in enumerate(texts) if text in case["golds"]]})
        statuses[status] += 1
    return groups, dict(statuses)


def encode(groups, vocab, window, torch):
    index = {char: i for i, char in enumerate(vocab) if len(char) == 1}
    sequences, offsets, bounds = [], [], []
    for group in groups:
        # Runtime scores only representable texts. OOV rows are placeholders, so they
        # must not shorten the context available to the actually scored candidates.
        texts = [text if known else group["texts"][0]
                 for text, known in zip(group["texts"], group["known_mask"], strict=True)]
        room = window - max(map(len, texts)) - 1
        if room < 0:
            raise ValueError("candidate does not fit model context")
        suffix = group["context"][-room:] if room else ""
        prefix = [2] + [index.get(char, 1) for char in suffix]
        start = len(sequences)
        for text in texts:
            sequences.append(prefix + [index.get(char, 1) for char in text])
            offsets.append((len(prefix) - 1, len(text)))
        bounds.append((start, len(sequences)))
    width = max(map(len, sequences))
    tokens = torch.zeros((len(sequences), width), dtype=torch.long)
    mask = torch.zeros((len(sequences), width - 1))
    for i, (sequence, (offset, length)) in enumerate(zip(sequences, offsets, strict=True)):
        tokens[i, :len(sequence)] = torch.tensor(sequence)
        mask[i, offset:offset + length] = 1
    return tokens, mask, bounds


def language_scores(model, tokens, mask, torch):
    inputs = tokens[:, :-1]
    positions = torch.arange(inputs.shape[1], device=inputs.device)
    x = model.drop(model.tok(inputs) + model.pos(positions))
    for block in model.blocks:
        x = block(x)
    x = model.ln_f(x)
    rows, positions = mask.nonzero(as_tuple=True)
    logits = model.head(x[rows, positions])
    targets = tokens[rows, positions + 1, None]
    logp = logits.float().log_softmax(-1).gather(-1, targets).squeeze(-1)
    return logp.new_zeros(tokens.shape[0]).index_add(0, rows, logp) / mask.sum(-1)


def ranking_loss(values, groups, bounds, torch):
    losses = []
    for group, (start, end) in zip(groups, bounds, strict=True):
        language = values[start:end]
        known = torch.tensor(group["known_mask"], dtype=torch.bool, device=values.device)
        language = torch.where(known, language, language[0])
        scores = language - values.new_tensor(group["penalties"])
        scores = scores + values.new_tensor([0.25] + [0.0] * (end - start - 1))
        # A single gold is ordinary CE; an accepted gold set uses its total probability.
        losses.append(scores.logsumexp(0) - scores[group["gold_indices"]].logsumexp(0))
    return torch.stack(losses).mean()


def export_model(model, metadata, parent_sha, binding, output, epoch, steps, torch, save_file):
    inherited = dict(metadata)
    inherited.update(parent_model_sha256=parent_sha, training_recipe_sha256=binding,
                     additional_optimizer_steps=str(steps), ranking_epoch=str(epoch))
    floats, quantized = {}, {}
    for name, parameter in model.state_dict().items():
        if name == "head.weight":
            continue
        value = parameter.detach().cpu().float().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}: nonfinite model weights")
        floats[name] = value
        if value.ndim == 2 and name != "pos.weight":
            scale = value.abs().amax(dim=1).clamp(min=1e-8) / 127
            quantized[name] = (value / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            quantized[name + ".scale"] = scale
        else:
            quantized[name] = value
    paths = {}
    for precision, tensors in [("f32", floats), ("int8", quantized)]:
        path = output / f"epoch{epoch:02d}-{precision}.safetensors"
        save_file(tensors, str(path), dict(inherited, precision=precision))
        paths[precision] = {"path": str(path), "sha256": digest(path)}
    return paths


def candidate_identity(candidate):
    return tuple(candidate[key] for key in ["text", "consumed", "source", "score_bits"])


def compare_reports(cases, before, after):
    old_rows = aligned_rows(cases, before)
    new_rows = aligned_rows(cases, after)
    rescues, breakages = [], []
    grouped = defaultdict(list)
    for case, old, new in zip(cases, old_rows, new_rows, strict=True):
        if (old["static_rank"] != new["static_rank"] or
                sorted(map(candidate_identity, old["candidates"])) !=
                sorted(map(candidate_identity, new["candidates"]))):
            raise ValueError(f"{case['id']}: ranking changed static candidate identity")
        grouped[case.get("group_id", case["input"])].append((case, new))
        change = {"id": case["id"], "before_rank": old["rank"], "after_rank": new["rank"]}
        if old["rank"] != 1 and new["rank"] == 1:
            rescues.append(change)
        elif old["rank"] == 1 and new["rank"] != 1:
            breakages.append(change)
    macro = sum((Fraction(sum(row["rank"] == 1 for _, row in members), len(members))
                 for members in grouped.values()), Fraction()) / len(grouped)
    multi = [members for members in grouped.values()
             if len({gold for case, _ in members for gold in case["golds"]}) > 1]
    return {"n": len(cases), "groups": len(grouped), "group_top1": float(macro),
            "group_top1_fraction": [macro.numerator, macro.denominator],
            "multi_target_groups": len(multi),
            "all_targets_top1_groups": sum(all(row["rank"] == 1 for _, row in members)
                                          for members in multi),
            "metrics": after["buckets"]["all"], "buckets": after["buckets"],
            "rescues": rescues, "breakages": breakages, "candidate_identity_unchanged": True}


def cli_evaluate(binary, data, output, model=None):
    command = [str(binary), "--dictionary-dir", str(DICTIONARIES)]
    if model is not None:
        command += ["--model", str(model)]
    command += ["evaluate", "--data", str(data), "--output", str(output)]
    started = time.monotonic()
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads(output.read_text(encoding="utf-8")), {
        "command": command, "seconds": time.monotonic() - started,
        "path": str(output), "sha256": digest(output)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "binary", "train", "dev", "output-dir"]:
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--epochs", type=int, default=3)
    result.add_argument("--lr", type=float, default=1e-5)
    result.add_argument("--seed", type=int, default=20261019)
    result.add_argument("--batch", type=int, default=4)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    result.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    result.add_argument("--text-train", type=Path,
                        help="add one ordinary text CE batch to every ranking update")
    result.add_argument("--text-batch", type=int, default=16)
    return result


def main():
    args = parser().parse_args()
    if args.epochs < 0 or min(args.batch, args.text_batch, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs must be nonnegative; batch, threads and lr must be positive")
    train, dev = read_cases(args.train), read_cases(args.dev)
    reject_overlap(train, dev)
    for key in ["model", "binary", "train", "dev", "output_dir"]:
        setattr(args, key, getattr(args, key).resolve())
    if args.text_train is not None:
        args.text_train = args.text_train.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors.torch import save_file
    import pretrain_model as text_training
    text_training.check_runtime(args.device, args.precision, torch)
    sys.path.insert(0, str(ROOT / "vendor/chinese-ime-training/neural"))
    from initialize import load_initial_model
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(args.device == "cpu")
    model, cfg, vocab, metadata = load_initial_model(args.model)
    model.to(args.device)
    model.eval()
    text_corpus = text_counts = None
    if args.text_train is not None:
        import numpy as np
        text_corpus, text_counts = text_training.encode_text(
            args.text_train, vocab, cfg.context, args.output_dir / "text-train.tokens.u16")
        text_rng = np.random.default_rng(args.seed)
        text_order, text_cursor = text_rng.permutation(len(text_corpus)), 0
    source_paths = sorted((ROOT / "src").rglob("*.rs")) + [
        ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__).resolve(),
        ROOT / "scripts/pretrain_model.py",
        ROOT / "scripts/training-requirements.txt", ROOT / "vendor/chinese-ime-lm/src/lib.rs",
        ROOT / "vendor/chinese-ime-training/neural/model.py",
        ROOT / "vendor/chinese-ime-training/neural/initialize.py"]
    recipe = {"format": "yanliu-ranker-training-v1", "arguments": {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch": torch.__version__, "config": vars(cfg), "model_metadata": metadata,
        "runtime": {"device": args.device, "precision": args.precision,
                    "attention_backend": "math" if args.precision == "bf16" else "framework_default",
                    "name": torch.xpu.get_device_name(0) if args.device == "xpu" else "CPU",
                    "parameter_dtype": "float32", "deterministic_algorithms": args.device == "cpu"},
        "initial_model_sha256": digest(args.model), "binary_sha256": digest(args.binary),
        "data": {"train": digest(args.train), "dev": digest(args.dev)},
        "source": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        "dictionaries": {name: digest(DICTIONARIES / name)
                         for name in ["8105.dict.yaml", "base.dict.yaml", "ext.dict.yaml"]},
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.01, "clip_norm": 1.0},
        "loss": ("rankCE + ordinary_text_CE; fixed coefficient 1 for each" if args.text_train else "rankCE"),
        "ranking_loss": "CE(mean_character_logp - direct_word_static_gap * 2 + leader_margin_0.25); gold-set marginal CE",
        "ranking_projection": "full-context Transformer and dropout; vocabulary head and log_softmax only at nonzero candidate mask positions; mean logp per candidate",
        "oov_policy": "match runtime: ASCII letters or candidates at least context-length neutralize the whole group; skip groups with OOV leader or fewer than two representable candidates; skip groups whose in-shortlist golds are all OOV because their leader-tied language score cannot clear the leader margin; retain the full shortlist and accepted gold identities otherwise; OOV language score equals the differentiable leader language score, leaving static penalties and promotion margin intact",
        "loss_reporting": "mean_loss and mean_rank_loss are group-weighted; text_cross_entropy is effective-target-weighted",
        "text_counts": text_counts,
        "selection": "actual CLI int8 development group-mean top1; ties choose earliest epoch",
        "learning": "system learning disabled; only explicitly supplied training files are read"}
    if args.text_train is not None:
        recipe["data"]["text_train"] = digest(args.text_train)
    write_json(args.output_dir / "recipe.json", recipe)
    recipe_sha = digest(args.output_dir / "recipe.json")
    train_report, train_command = cli_evaluate(args.binary, args.train, args.output_dir / "train-static.json")
    groups, statuses = training_groups(train, train_report, cfg.context, vocab)
    write_json(args.output_dir / "training-data.json", {
        "n": len(train), "trainable": len(groups), "statuses": statuses,
        "static_cli": train_command, "groups": groups})
    del train_report, train
    if not groups and args.epochs:
        raise ValueError("no trainable candidate groups; see training-data.json")
    initial_artifacts = export_model(model, metadata, recipe["initial_model_sha256"], recipe_sha,
                                     args.output_dir, 0, 0, torch, save_file)
    initial_report, initial_command = cli_evaluate(
        args.binary, args.dev, args.output_dir / "dev-initial.json", initial_artifacts["int8"]["path"])
    if metadata.get("precision") == "int8":
        source_report, _ = cli_evaluate(args.binary, args.dev, args.output_dir / "dev-source.json", args.model)
        if source_report["rows"] != initial_report["rows"]:
            raise ValueError("epoch0 int8 roundtrip changed original-model CLI rows")
    aligned_rows(dev, initial_report)
    initial = compare_reports(dev, initial_report, initial_report)
    write_json(args.output_dir / "development-initial.json", dict(initial, cli=initial_command))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                 weight_decay=0.01, foreach=False)
    steps, training, development = 0, [], []
    for epoch in range(args.epochs + 1):
        if epoch:
            started = time.monotonic()
            model.train(args.text_train is not None)
            order = list(range(len(groups)))
            random.Random(args.seed + epoch).shuffle(order)
            total = rank_total = text_total = 0.0
            text_tokens = 0
            for offset in range(0, len(order), args.batch):
                batch = [groups[index] for index in order[offset:offset + args.batch]]
                tokens, mask, bounds = encode(batch, vocab, cfg.context, torch)
                tokens, mask = tokens.to(args.device), mask.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                with text_training.autocast(args.device, args.precision, torch):
                    rank_loss = ranking_loss(language_scores(model, tokens, mask, torch), batch, bounds, torch)
                    loss = rank_loss
                    if text_corpus is not None:
                        if text_cursor == len(text_order):
                            text_order, text_cursor = text_rng.permutation(len(text_corpus)), 0
                        indices = text_order[text_cursor:text_cursor + args.text_batch]
                        text_cursor += len(indices)
                        inputs, targets, n = text_training.text_batch(text_corpus, indices, args.device, torch)
                        _, text_loss = model(inputs, targets)
                        loss = rank_loss + text_loss
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss at epoch {epoch}, batch {offset // args.batch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                total += float(loss.detach()) * len(batch)
                rank_total += float(rank_loss.detach()) * len(batch)
                if text_corpus is not None:
                    text_total += float(text_loss.detach()) * n
                    text_tokens += n
                if steps % 100 == 0:
                    print(json.dumps({"epoch": epoch, "optimizer_steps": steps,
                                      "groups_seen": offset + len(batch), "groups_total": len(groups),
                                      "mean_loss": total / (offset + len(batch)),
                                      "mean_rank_loss": rank_total / (offset + len(batch)),
                                      "text_cross_entropy": text_total / text_tokens if text_tokens else None,
                                      "text_effective_targets": text_tokens,
                                      "seconds": time.monotonic() - started}), flush=True)
            training.append({"epoch": epoch, "n": len(groups), "mean_loss": total / len(groups),
                             "mean_rank_loss": rank_total / len(groups),
                             "text_cross_entropy": text_total / text_tokens if text_tokens else None,
                             "text_effective_targets": text_tokens,
                             "optimizer_steps": steps, "seconds": time.monotonic() - started})
        if epoch:
            artifacts = export_model(model, metadata, recipe["initial_model_sha256"], recipe_sha,
                                     args.output_dir, epoch, steps, torch, save_file)
            report, command = cli_evaluate(args.binary, args.dev,
                args.output_dir / f"dev-epoch{epoch:02d}.json", artifacts["int8"]["path"])
        else:
            artifacts, report, command = initial_artifacts, initial_report, initial_command
        comparison = compare_reports(dev, initial_report, report)
        record = dict(comparison, epoch=epoch, optimizer_steps=steps, artifacts=artifacts, cli=command)
        development.append(record)
        write_json(args.output_dir / f"development-epoch{epoch:02d}.json", record)
        print(json.dumps({"epoch": epoch, "mean_loss": training[-1]["mean_loss"] if epoch else None,
                          "group_top1": comparison["group_top1"], "metrics": comparison["metrics"]}), flush=True)
    for path, expected in recipe["data"].items():
        if digest(getattr(args, path)) != expected:
            raise ValueError(f"{path} input changed during training")
    if digest(args.binary) != recipe["binary_sha256"] or digest(args.model) != recipe["initial_model_sha256"]:
        raise ValueError("binary or initial model changed during training")
    for relative, expected in recipe["source"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"source changed during training: {relative}")
    for name, expected in recipe["dictionaries"].items():
        if digest(DICTIONARIES / name) != expected:
            raise ValueError(f"dictionary changed during training: {name}")
    write_json(args.output_dir / "training.json", training)
    write_json(args.output_dir / "development.json", development)
    selected = max(development, key=lambda record: Fraction(*record["group_top1_fraction"]))
    selection = {"epoch": selected["epoch"], "model": selected["artifacts"]["int8"],
                 "group_top1_fraction": selected["group_top1_fraction"],
                 "initial": initial, "selected": selected,
                 "recipe_sha256": recipe_sha, "installation": "not changed"}
    write_json(args.output_dir / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
