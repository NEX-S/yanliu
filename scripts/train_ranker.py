#!/usr/bin/env python3
"""Fine-tune the current character model on real CLI candidate shortlists.

Install scripts/training-requirements.txt in a private virtual environment.
Only explicitly supplied train/dev JSONL and optional text files are read. The fresh output
directory retains every development case and all epoch artifacts; this command
does not install a model or access system learning.
"""
import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
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


def is_correction(source):
    if source in {"ExactDictionary", "Composed", "Completion", "User"}:
        return False
    if source.startswith(("MissingLetter {", "NeighborSubstitution {", "Transposition {")):
        return True
    raise ValueError(f"unknown CLI candidate source: {source!r}")


def aligned_rows(cases, report):
    if len(cases) != len(report["rows"]):
        raise ValueError("CLI report changed the case denominator")
    for case, row in zip(cases, report["rows"], strict=True):
        if case["id"] != row["id"]:
            raise ValueError(f"CLI case order/id mismatch: {case['id']} != {row['id']}")
    return report["rows"]


def training_groups(cases, report, vocab, window):
    groups, statuses = [], Counter()
    characters = set(vocab[3:])
    for case, row in zip(cases, aligned_rows(cases, report), strict=True):
        scoring = row["scoring"]
        if scoring is None:
            raise ValueError(f"{case['id']}: source model scoring is required")
        candidates, promotion = scoring["candidates"], scoring["promotion"]
        scores = promotion["scores"]
        indices = [score["index"] for score in scores]
        if (any(type(index) is not int or not 0 <= index < len(candidates) for index in indices)
                or len(set(indices)) != len(indices) or (indices and indices[0] != 0)):
            raise ValueError(f"{case['id']}: invalid CLI scoring indices")
        if any(candidates[index]["consumed"] != len(case["input"].encode("utf-8"))
               or candidates[index]["source"] == "Completion" for index in indices):
            raise ValueError(f"{case['id']}: scoring requires complete conversion candidates")
        if (any(not math.isfinite(score["dictionary"]) or score["dictionary"] > 0
                or not math.isfinite(score["language"]) for score in scores)
                or not math.isfinite(promotion["margin"]) or promotion["margin"] < 0):
            raise ValueError(f"{case['id']}: invalid CLI scoring contribution")
        texts = [candidates[index]["text"] for index in indices]
        known = [all(char in characters for char in text) for text in texts]
        if len(indices) < 2 or len(set(texts)) < 2:
            status = "protected_or_no_competition"
        elif not known[0]:
            status = "neutral_language_group_oov_leader"
        elif sum(known) < 2:
            status = "neutral_language_group_insufficient_known_candidates"
        elif all(score["language"] == 0.0 for score in scores):
            status = "neutral_language_group"
        elif not any(text in case["golds"] for text in texts):
            status = "gold_outside_scoring"
            targets = [gold for gold in case["golds"]
                       if len(gold) < window and all(char in characters for char in gold)
                       and not any(char.isascii() and char.isalpha() for char in gold)]
            if targets:
                groups.append({"id": case["id"], "context": case["context"], "texts": targets,
                               "known_mask": [True] * len(targets), "target_only": True})
        elif not any(is_known and text in case["golds"]
                     for text, is_known in zip(texts, known, strict=True)):
            status = "gold_only_unrepresented"
        elif all(text in case["golds"] for text in texts):
            status = "no_negative_candidate"
        else:
            status = "trainable"
            groups.append({"id": case["id"], "context": case["context"], "texts": texts,
                           "source_indices": indices, "known_mask": known,
                           "composed_leader": candidates[0]["source"] == "Composed",
                           "penalties": [-score["dictionary"] for score in scores],
                           "margin": promotion["margin"],
                           "gold_indices": [i for i, text in enumerate(texts) if text in case["golds"]]})
            if row["rank"] == 1:
                groups[-1]["retain_top1"] = True
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


def training_losses(values, groups, bounds, torch):
    losses = []
    for group, (start, end) in zip(groups, bounds, strict=True):
        language = values[start:end]
        if group.get("target_only", False):
            losses.append(-language.mean())
            continue
        if group["composed_leader"]:
            language = language * values.new_tensor([len(text) / len(group["texts"][0])
                                                    for text in group["texts"]])
        known = torch.tensor(group["known_mask"], dtype=torch.bool, device=values.device)
        language = torch.where(known, language, language[0])
        scores = language - values.new_tensor(group["penalties"])
        scores = scores + values.new_tensor([group["margin"]] + [0.0] * (end - start - 1))
        if group.get("retain_top1", False):
            non_gold = [i for i in range(end - start) if i not in group["gold_indices"]]
            losses.append(torch.relu(group["margin"] + scores[non_gold].max()
                                     - scores[group["gold_indices"]].max()))
        else:
            # A single gold is ordinary CE; an accepted gold set uses its total probability.
            losses.append(scores.logsumexp(0) - scores[group["gold_indices"]].logsumexp(0))
    return torch.stack(losses)


def text_teacher_kl(student_logits, teacher_logits, targets, torch):
    valid = targets.ne(0)
    return torch.nn.functional.kl_div(
        student_logits[valid].float().log_softmax(-1),
        teacher_logits[valid].float().softmax(-1), reduction="batchmean")


def export_model(model, metadata, runtime, parent_sha, binding, output, epoch, steps, torch, save_file):
    inherited = dict(metadata)
    inherited.update(parent_model_sha256=parent_sha, training_recipe_sha256=binding,
                     additional_optimizer_steps=str(steps), ranking_epoch=str(epoch),
                     training_device=runtime["device"], training_precision=runtime["precision"])
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


def cli_behavior(report):
    rows = []
    for row in report["rows"]:
        scoring = row["scoring"]
        considered = None if scoring is None else (
            list(map(candidate_identity, scoring["candidates"])),
            [score["index"] for score in scoring["promotion"]["scores"]],
            scoring["promotion"]["index"])
        rows.append((row["id"], row["rank"], row["static_rank"],
                     list(map(candidate_identity, row["candidates"])), considered))
    return rows


def candidate_additions(case, static, row):
    old = Counter(map(candidate_identity, static["candidates"]))
    new = Counter(map(candidate_identity, row["candidates"]))
    additions = new - old
    replacements = {identity[:2] for identity in additions}
    ordinary = {identity[:2] for identity in old
                if identity[2] != "Completion" and not is_correction(identity[2])}
    if (static["static_rank"] != row["static_rank"] or
            any(consumed != len(case["input"].encode("utf-8")) or source != "Composed"
                or (text, consumed) in ordinary
                for text, consumed, source, _ in additions) or
            any((source != "Completion" and not is_correction(source))
                or (text, consumed) not in replacements
                for text, consumed, source, _ in old - new)):
        raise ValueError(f"{case['id']}: ranking changed static candidate identity")
    return additions.total()


def compare_reports(cases, before, after, static):
    old_rows = aligned_rows(cases, before)
    new_rows = aligned_rows(cases, after)
    static_rows = aligned_rows(cases, static)
    rescues, breakages = [], []
    grouped = defaultdict(list)
    identity_unchanged, added_candidates = True, 0
    for case, old, new, baseline in zip(cases, old_rows, new_rows, static_rows, strict=True):
        candidate_additions(case, baseline, old)
        added = candidate_additions(case, baseline, new)
        identity_unchanged &= added == 0
        added_candidates += added
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
            "rescues": rescues, "breakages": breakages,
            "candidate_identity_unchanged": identity_unchanged,
            "static_candidate_identity_preserved": True, "added_candidates": added_candidates}


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
                        help="add one frozen-source teacher KL text batch to every ranking update")
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
    teacher = copy.deepcopy(model).eval().requires_grad_(False) if args.text_train is not None else None
    text_corpus = text_counts = None
    if args.text_train is not None:
        import numpy as np
        text_corpus, text_counts = text_training.encode_text(
            args.text_train, vocab, cfg.context, args.output_dir / "text-train.tokens.u16")
        text_rng = np.random.default_rng(args.seed)
        text_order, text_cursor = text_rng.permutation(len(text_corpus)), 0
    source_paths = (sorted((ROOT / "src").rglob("*.rs"))
                    + sorted((ROOT / "vendor/chinese-ime-lm/src").rglob("*.rs"))) + [
        ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__).resolve(),
        ROOT / "scripts/pretrain_model.py",
        ROOT / "scripts/training-requirements.txt",
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
        "loss": "mean supervised group loss (source-top1 retention hinge, otherwise real-menu rank CE; missing-target suffix CE)"
                + (" + mean effective-token KL(frozen_source_teacher || student), T=1; fixed coefficient 1 for each" if args.text_train else ""),
        "text_teacher": {"initialization": "deepcopy of initialized source model; no second random initialization",
                         "mode": "eval; requires_grad=False; forward under torch.no_grad",
                         "objective": "KL(teacher || student) over the full vocabulary at targets != PAD(0); UNK targets are already masked to PAD, UNK context is retained",
                         "temperature": 1, "coefficient": 1,
                         "reduction": "batchmean after selecting effective target positions; denominator is effective tokens",
                         "student_mode": "original train mode and dropout; no additional ordinary text CE"}
                        if args.text_train is not None else None,
        "ranking_loss": "ordinary rank groups whose actual full source CLI row.rank == 1 use retention_loss; all other ordinary rank groups retain gold-set CE(leader_length_normalized_logp + actual source-model dictionary contribution + actual leader promotion margin); all development cases retained",
        "retention_loss": "relu(actual promotion margin + max(non-gold adjusted scores) - max(accepted-gold adjusted scores)); adjusted scores keep the original known-mask leader-tied language, dictionary contribution and leader promotion margin. The extra margin is a protection buffer, not an equivalent rewrite of runtime promotion: a nonleader gold against a wrong original leader needs an unadjusted score gap of at least twice the margin. Safely retained groups have zero loss and gradient; accepted golds use their best score, not their summed probability.",
        "ranking_candidates": "source-model CLI Model::refine scoring.candidates and promotion.scores indices, including bounded neural recall; no Python shortlist or static-score reconstruction",
        "missing_target_loss": "For representable, full-context-fitting golds absent from actual scoring, teacher-force only the gold suffix; negative mean character logp, accepted golds independently equally weighted; supervision is a separate target-only group, never a CLI candidate. Rank and target-only groups are equally weighted in the supervised group mean.",
        "ranking_projection": "full-context Transformer and dropout; vocabulary head and log_softmax only at nonzero candidate mask positions; mean logp per candidate",
        "ranking_normalization": "only for an original CLI Composed leader, multiply candidate mean character logp by candidate character count / leader character count before tying OOV to leader; other leaders and same-length scores unchanged; target-only suffix CE retains mean character logp",
        "oov_policy": "use actual CLI neutral-language groups; skip groups with OOV leader or fewer than two representable candidates; skip groups whose scored golds are all OOV because their leader-tied language score cannot clear the leader margin; retain actual source-model considered candidates and accepted gold identities otherwise; OOV language score equals the differentiable leader language score, leaving actual dictionary contributions and promotion margin intact",
        "loss_reporting": "mean_loss is weighted by all supervised groups; mean_rank_loss combines retention hinge and remaining rank CE over all observed ordinary rank groups; mean_target_only_loss uses its own observed group count; text_teacher_kl is effective-target-weighted",
        "text_counts": text_counts,
        "selection": "actual CLI int8 development group-mean top1; ties choose earliest epoch",
        "learning": "system learning disabled; only explicitly supplied training files are read"}
    if args.text_train is not None:
        recipe["data"]["text_train"] = digest(args.text_train)
    write_json(args.output_dir / "recipe.json", recipe)
    recipe_sha = digest(args.output_dir / "recipe.json")
    train_report, train_command = cli_evaluate(
        args.binary, args.train, args.output_dir / "train-source.json", args.model)
    groups, statuses = training_groups(train, train_report, vocab, cfg.context)
    target_groups = sum(group.get("target_only", False) for group in groups)
    retention_groups = sum(group.get("retain_top1", False) for group in groups)
    write_json(args.output_dir / "training-data.json", {
        "n": len(train), "trainable": len(groups), "rank_groups": len(groups) - target_groups,
        "retention_groups": retention_groups,
        "rank_ce_groups": len(groups) - target_groups - retention_groups,
        "target_only_groups": target_groups,
        "target_only_unrepresented": statuses.get("gold_outside_scoring", 0) - target_groups,
        "statuses": statuses,
        "source_model_cli": train_command, "groups": groups})
    del train_report, train
    if not groups and args.epochs:
        raise ValueError("no trainable candidate groups; see training-data.json")
    initial_artifacts = export_model(model, metadata, recipe["runtime"], recipe["initial_model_sha256"],
                                     recipe_sha, args.output_dir, 0, 0, torch, save_file)
    initial_report, initial_command = cli_evaluate(
        args.binary, args.dev, args.output_dir / "dev-initial.json", initial_artifacts["int8"]["path"])
    static_report, static_command = cli_evaluate(
        args.binary, args.dev, args.output_dir / "dev-static.json")
    if metadata.get("precision") == "int8":
        source_report, _ = cli_evaluate(args.binary, args.dev, args.output_dir / "dev-source.json", args.model)
        if cli_behavior(source_report) != cli_behavior(initial_report):
            raise ValueError("epoch0 int8 roundtrip changed original-model CLI rows")
    aligned_rows(dev, initial_report)
    initial = compare_reports(dev, initial_report, initial_report, static_report)
    write_json(args.output_dir / "development-initial.json",
               dict(initial, cli=initial_command, static_cli=static_command))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                 weight_decay=0.01, foreach=False)
    steps, training, development = 0, [], []
    for epoch in range(args.epochs + 1):
        if epoch:
            started = time.monotonic()
            model.train(args.text_train is not None)
            order = list(range(len(groups)))
            random.Random(args.seed + epoch).shuffle(order)
            total = rank_total = target_total = text_total = 0.0
            rank_seen = target_seen = text_tokens = 0
            for offset in range(0, len(order), args.batch):
                batch = [groups[index] for index in order[offset:offset + args.batch]]
                tokens, mask, bounds = encode(batch, vocab, cfg.context, torch)
                tokens, mask = tokens.to(args.device), mask.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                with text_training.autocast(args.device, args.precision, torch):
                    losses = training_losses(language_scores(model, tokens, mask, torch), batch, bounds, torch)
                    loss = losses.mean()
                    if text_corpus is not None:
                        if text_cursor == len(text_order):
                            text_order, text_cursor = text_rng.permutation(len(text_corpus)), 0
                        indices = text_order[text_cursor:text_cursor + args.text_batch]
                        text_cursor += len(indices)
                        inputs, targets, n = text_training.text_batch(text_corpus, indices, args.device, torch)
                        student_logits, _ = model(inputs)
                        with torch.no_grad():
                            teacher_logits, _ = teacher(inputs)
                        text_loss = text_teacher_kl(student_logits, teacher_logits, targets, torch)
                        loss = loss + text_loss
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss at epoch {epoch}, batch {offset // args.batch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                total += float(loss.detach()) * len(batch)
                rank_indices = [i for i, group in enumerate(batch) if not group.get("target_only", False)]
                target_indices = [i for i, group in enumerate(batch) if group.get("target_only", False)]
                rank_total += float(losses[rank_indices].detach().sum())
                target_total += float(losses[target_indices].detach().sum())
                rank_seen += len(rank_indices)
                target_seen += len(target_indices)
                if text_corpus is not None:
                    text_total += float(text_loss.detach()) * n
                    text_tokens += n
                if steps % 100 == 0:
                    print(json.dumps({"epoch": epoch, "optimizer_steps": steps,
                                      "groups_seen": offset + len(batch), "groups_total": len(groups),
                                      "mean_loss": total / (offset + len(batch)),
                                      "mean_rank_loss": rank_total / rank_seen if rank_seen else None,
                                      "rank_groups_seen": rank_seen,
                                      "mean_target_only_loss": target_total / target_seen if target_seen else None,
                                      "target_only_groups_seen": target_seen,
                                      "text_teacher_kl": text_total / text_tokens if text_tokens else None,
                                      "text_effective_targets": text_tokens,
                                      "seconds": time.monotonic() - started}), flush=True)
            training.append({"epoch": epoch, "n": len(groups), "mean_loss": total / len(groups),
                             "mean_rank_loss": rank_total / rank_seen if rank_seen else None,
                             "rank_groups": rank_seen, "target_only_groups": target_seen,
                             "mean_target_only_loss": target_total / target_seen if target_seen else None,
                             "text_teacher_kl": text_total / text_tokens if text_tokens else None,
                             "text_effective_targets": text_tokens,
                             "optimizer_steps": steps, "seconds": time.monotonic() - started})
        if epoch:
            artifacts = export_model(model, metadata, recipe["runtime"], recipe["initial_model_sha256"],
                                     recipe_sha, args.output_dir, epoch, steps, torch, save_file)
            report, command = cli_evaluate(args.binary, args.dev,
                args.output_dir / f"dev-epoch{epoch:02d}.json", artifacts["int8"]["path"])
        else:
            artifacts, report, command = initial_artifacts, initial_report, initial_command
        comparison = compare_reports(dev, initial_report, report, static_report)
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
