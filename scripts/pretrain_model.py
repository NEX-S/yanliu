#!/usr/bin/env python3
"""Train random character-model weights on explicitly supplied paragraph text.

Each UTF-8 line is an independent paragraph. --spec supplies the model shape,
character vocabulary and provenance; no pretrained weights are loaded.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_spec(path):
    with Path(path).open(encoding="utf-8") as stream:
        spec = json.load(stream)
    if not isinstance(spec, dict) or set(spec) != {"config", "vocab", "provenance"}:
        raise ValueError(f"{path}: expected config, vocab and provenance")
    cfg, vocab, provenance = spec["config"], spec["vocab"], spec["provenance"]
    dimensions = ("vocab", "n_layer", "n_head", "n_embd", "context")
    if (not isinstance(cfg, dict) or set(cfg) != set(dimensions) | {"dropout"}
            or any(type(cfg.get(key)) is not int or cfg[key] < 1
                                       for key in dimensions)
            or not 4 <= cfg["vocab"] <= 65535 or cfg["n_embd"] % cfg["n_head"]
            or cfg["context"] != 64
            or type(cfg["dropout"]) not in (int, float)
            or not math.isfinite(cfg["dropout"]) or not 0 <= cfg["dropout"] < 1
            or not isinstance(vocab, list) or len(vocab) != cfg["vocab"]
            or vocab[:3] != ["<pad>", "<unk>", "<bos>"]
            or any(not isinstance(char, str) or len(char) != 1
                   or 0xD800 <= ord(char) <= 0xDFFF for char in vocab[3:])
            or len(set(vocab)) != len(vocab)):
        raise ValueError(f"{path}: invalid character vocabulary/model configuration")
    if (not isinstance(provenance, dict)
            or any(not isinstance(provenance.get(key), str) or not provenance[key].strip()
                   for key in ("license", "data_classification", "attribution"))):
        raise ValueError(f"{path}: provenance requires license, data_classification and attribution")
    return cfg, vocab, provenance


def initialize_random(config, torch, seed):
    sys.path.insert(0, str(ROOT / "vendor/chinese-ime-training/neural"))
    from model import CharLM, Config
    torch.manual_seed(seed)
    return CharLM(Config(**config))


def check_runtime(device, precision, torch):
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU training requires explicit fp32 precision")
    if device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU was requested but is unavailable; no device fallback")


@contextmanager
def autocast(device, precision, torch):
    if precision == "fp32":
        yield
        return
    # XPU's fused BF16 SDPA produced intermittent NaNs with finite Q/K/V inputs.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), \
            torch.autocast(device_type=device, dtype=torch.bfloat16):
        yield


def encode_text(path, vocab, window, destination):
    """Keep UNK context, but emit only blocks with at least one known target."""
    import numpy as np
    index = {char: number for number, char in enumerate(vocab) if len(char) == 1}
    paragraphs = blocks = targets = unknown = empty_lines = 0
    skipped = masked_unknown = padding = 0
    with Path(path).open(encoding="utf-8") as source, Path(destination).open("xb") as output:
        for line in source:
            text = line.rstrip("\r\n")
            if not text:
                empty_lines += 1
                continue
            encoded = [2] + [index.get(char, 1) for char in text]
            paragraphs += 1
            targets += sum(token > 1 for token in encoded[1:])
            unknown += encoded.count(1)
            for start in range(0, len(encoded) - 1, window):
                piece = encoded[start:start + window + 1]
                if not any(token > 1 for token in piece[1:]):
                    skipped += 1
                    continue
                block = np.zeros(window + 1, dtype="<u2")
                block[:len(piece)] = piece
                output.write(block.tobytes())
                blocks += 1
                masked_unknown += piece[1:].count(1)
                padding += window + 1 - len(piece)
    if not blocks:
        raise ValueError(f"{path}: no effective training targets in the supplied vocabulary")
    corpus = np.memmap(destination, dtype="<u2", mode="r", shape=(blocks, window + 1))
    return corpus, {"paragraphs": paragraphs, "empty_lines": empty_lines, "blocks": blocks,
                    "effective_targets": targets, "unknown_targets": unknown,
                    "source_targets": targets + unknown, "masked_unknown_targets": masked_unknown,
                    "skipped_no_target_blocks": skipped, "padding_targets": padding}


def text_batch(corpus, indices, device, torch, token_limit=None):
    import numpy as np
    blocks = np.array(corpus[indices], dtype="int64", copy=True)
    span = np.count_nonzero(blocks[:, 1:], axis=1).max()
    tokens = torch.from_numpy(blocks[:, :span + 1])
    inputs, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].clone()
    targets[targets == 1] = 0
    valid = targets.flatten().nonzero().flatten()
    if token_limit is not None and len(valid) > token_limit:
        targets.flatten()[valid[token_limit:]] = 0
        count = token_limit
    else:
        count = len(valid)
    if not count:
        raise ValueError("text batch has no effective targets")
    return inputs.to(device), targets.to(device), count


def length_groups(corpus):
    import numpy as np
    lengths = np.count_nonzero(corpus[:, 1:], axis=1)
    return [(length, np.flatnonzero(lengths == length))
            for length in range(1, corpus.shape[1])]


def text_batches(groups, position_budget, rng=None):
    batches = []
    for length, indices in groups:
        order = rng.permutation(indices) if rng is not None else indices
        size = position_budget // length
        batches.extend(order[start:start + size] for start in range(0, len(order), size))
    if rng is not None:
        return [batches[index] for index in rng.permutation(len(batches))]
    return batches


def evaluate_text(model, corpus, batches, device, precision, torch):
    was_training = model.training
    model.eval()
    total = count = 0
    with torch.inference_mode():
        for number, indices in enumerate(batches, 1):
            inputs, targets, n = text_batch(corpus, indices, device, torch)
            with autocast(device, precision, torch):
                loss = model(inputs, targets)[1]
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError(f"nonfinite development text loss at batch {number}, effective_targets={n}")
            total += value * n
            count += n
    model.train(was_training)
    return {"cross_entropy": total / count, "effective_targets": count}


def save_best(model, metadata, path, torch, save_file):
    tensors = {}
    for name, parameter in model.state_dict().items():
        if name == "head.weight":
            continue
        value = parameter.detach().cpu().float().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}: nonfinite model weights")
        tensors[name] = value
    temporary = path.with_suffix(".tmp")
    save_file(tensors, str(temporary), metadata)
    temporary.replace(path)


def learning_rate(seen_targets, planned_targets, position_budget, peak):
    warmup = min(planned_targets * 0.05, 100 * position_budget)
    if seen_targets < warmup:
        return peak * seen_targets / warmup
    progress = min(1.0, (seen_targets - warmup) / (planned_targets - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "dev", "spec", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    result.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    result.add_argument("--epochs", type=int, default=12)
    result.add_argument("--max-tokens", type=int, default=300_000_000)
    result.add_argument("--max-seconds", type=float, default=6 * 60 * 60)
    result.add_argument("--batch", type=int, default=16,
                        help="token-position budget in equivalent 64-target windows per update")
    result.add_argument("--lr", type=float, default=3e-4)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--seed", type=int, default=20261020)
    return result


def main():
    args = parser().parse_args()
    if (min(args.epochs, args.max_tokens, args.batch, args.threads) < 1
            or not math.isfinite(args.max_seconds) or args.max_seconds <= 0
            or not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("epochs, token/time budgets, batch, threads and lr must be positive and finite")
    for key in ("train", "dev", "spec", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    data_hashes = {"train": digest(args.train), "dev": digest(args.dev)}
    if args.train == args.dev or data_hashes["train"] == data_hashes["dev"]:
        raise ValueError("train/dev text must be different files with different contents")
    config, vocab, provenance = read_spec(args.spec)
    spec_sha = digest(args.spec)
    import numpy as np
    import torch
    from safetensors.torch import save_file
    check_runtime(args.device, args.precision, torch)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(args.device == "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train, train_counts = encode_text(args.train, vocab, config["context"], args.output_dir / "train.tokens.u16")
    dev, dev_counts = encode_text(args.dev, vocab, config["context"], args.output_dir / "dev.tokens.u16")
    position_budget = args.batch * config["context"]
    planned_targets = min(args.max_tokens, train_counts["effective_targets"] * args.epochs)
    train_groups = length_groups(train)
    dev_batches = text_batches(length_groups(dev), position_budget)
    sources = [Path(__file__).resolve(), ROOT / "vendor/chinese-ime-training/neural/model.py"]
    recipe = {"format": "yanliu-random-pretraining-v3", "arguments": {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch": torch.__version__, "config": config, "initialization": "random; no parent weights loaded",
        "runtime": {"device": args.device, "precision": args.precision,
                    "attention_backend": "math" if args.precision == "bf16" else "framework_default",
                    "name": torch.xpu.get_device_name(0) if args.device == "xpu" else "CPU",
                    "parameter_dtype": "float32", "deterministic_algorithms": args.device == "cpu"},
        "spec": {"path": str(args.spec), "sha256": spec_sha},
        "data": data_hashes,
        "counts": {"train": train_counts, "dev": dev_counts},
        "source": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.01,
                      "clip_norm": 1.0, "main_parameter_precision": "float32",
                      "planned_effective_targets": planned_targets,
                      "warmup_effective_targets": min(planned_targets * 0.05, 100 * position_budget),
                      "schedule": "cumulative effective targets including the current update; warmup then cosine to 10% peak LR; peak LR is not scaled with batch size"},
        "batching": {"position_budget": position_budget, "batch_argument": "equivalent 64-target windows",
                     "method": "group blocks by non-PAD target span 1..64; shuffle within each group; batch size=floor(position_budget/span); shuffle batches each epoch; crop each batch to its non-PAD span; each block occurs once per complete epoch; development uses the same grouping without shuffling"},
        "sampling": "each paragraph separately; nonoverlapping target chunks, previous token retained at chunk boundary; no cross-line concatenation; PAD0 and UNK1 targets ignored; UNK1 retained in inputs; blocks without known targets omitted",
        "limits": "stop at first epoch/token/wall-time budget; evaluate after final partial epoch; stop after two consecutive dev CE non-improvements",
        "provenance": provenance,
        "selection": "lowest complete text dev next-character CE; ties keep earlier checkpoint"}
    write_json(args.output_dir / "recipe.json", recipe)
    recipe_sha = digest(args.output_dir / "recipe.json")
    model = initialize_random(config, torch, args.seed).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                 weight_decay=0.01, foreach=False)
    metadata = {"format": "chinese-ime-lm", "version": "1", "tied_embeddings": "true",
                "config": json.dumps(config), "vocab": json.dumps(vocab, ensure_ascii=False),
                "precision": "f32", "initialization": "random", "seed": str(args.seed),
                "training_device": args.device, "training_precision": args.precision,
                "license": provenance["license"], "data_classification": provenance["data_classification"],
                "attribution": provenance["attribution"],
                "provenance": json.dumps(provenance, ensure_ascii=False, allow_nan=False),
                "training_spec_sha256": spec_sha, "training_recipe_sha256": recipe_sha,
                "train_sha256": recipe["data"]["train"], "dev_sha256": recipe["data"]["dev"]}
    started = time.monotonic()
    best = evaluate_text(model, dev, dev_batches, args.device, args.precision, torch)
    best.update(epoch=0, optimizer_steps=0, training_effective_targets=0)
    best_path = args.output_dir / "best-f32.safetensors"
    save_best(model, dict(metadata, training_steps="0", training_effective_targets="0",
                          validation_loss=str(best["cross_entropy"])), best_path, torch, save_file)
    records = [dict(best)]
    steps = seen = stale = 0
    stop = "epoch_limit"
    for epoch in range(1, args.epochs + 1):
        batches = text_batches(train_groups, position_budget, np.random.default_rng(args.seed + epoch))
        model.train()
        epoch_total = epoch_tokens = ignored = 0
        epoch_started = time.monotonic()
        for indices in batches:
            if seen >= args.max_tokens:
                stop = "token_limit"
                break
            if time.monotonic() - started >= args.max_seconds:
                stop = "time_limit"
                break
            inputs, targets, n = text_batch(train, indices, args.device,
                                           torch, token_limit=args.max_tokens - seen)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(seen + n, planned_targets, position_budget, args.lr)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args.device, args.precision, torch):
                loss = model(inputs, targets)[1]
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError(f"nonfinite text loss at optimizer step {steps}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            seen += n
            epoch_tokens += n
            ignored += targets.numel() - n
            epoch_total += value * n
            if steps % 100 == 0:
                elapsed = time.monotonic() - epoch_started
                print(json.dumps({"epoch": epoch, "optimizer_steps": steps, "effective_targets": seen,
                                  "epoch_cross_entropy": epoch_total / epoch_tokens,
                                  "effective_targets_per_second": epoch_tokens / elapsed,
                                  "seconds": time.monotonic() - started}), flush=True)
        train_seconds = time.monotonic() - epoch_started
        current = evaluate_text(model, dev, dev_batches, args.device, args.precision, torch)
        current.update(epoch=epoch, optimizer_steps=steps, training_effective_targets=seen,
                       train_cross_entropy=epoch_total / epoch_tokens if epoch_tokens else None,
                       epoch_effective_targets=epoch_tokens, epoch_ignored_targets=ignored,
                       seconds=time.monotonic() - started,
                       training_seconds=train_seconds,
                       effective_targets_per_second=epoch_tokens / train_seconds)
        records.append(current)
        write_json(args.output_dir / f"epoch{epoch:02d}.json", current)
        print(json.dumps(current), flush=True)
        if current["cross_entropy"] < best["cross_entropy"]:
            best, stale = dict(current), 0
            save_best(model, dict(metadata, training_steps=str(steps), training_effective_targets=str(seen),
                                  validation_loss=str(current["cross_entropy"])), best_path, torch, save_file)
        else:
            stale += 1
        if stop in ("token_limit", "time_limit") or seen >= args.max_tokens:
            if seen >= args.max_tokens:
                stop = "token_limit"
            break
        if stale == 2:
            stop = "development_early_stop"
            break
    for key, expected in recipe["data"].items():
        if digest(getattr(args, key)) != expected:
            raise ValueError(f"{key} changed during pretraining")
    if digest(args.spec) != spec_sha:
        raise ValueError("training spec changed during pretraining")
    for relative, expected in recipe["source"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"source changed during pretraining: {relative}")
    write_json(args.output_dir / "training.json", records)
    selection = {"best": best, "model": {"path": str(best_path), "sha256": digest(best_path)},
                 "optimizer_steps": steps, "training_effective_targets": seen,
                 "seconds": time.monotonic() - started, "stop_reason": stop,
                 "recipe_sha256": recipe_sha, "installation": "not changed"}
    write_json(args.output_dir / "selection.json", selection)
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
