"""Spec boundaries, known-target masking and random training on a tiny CPU model."""
import importlib.util
from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/training-tests"
CACHE.mkdir(parents=True, exist_ok=True)
SPEC = importlib.util.spec_from_file_location("pretrain_model", ROOT / "scripts/pretrain_model.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)

try:
    import torch
    from safetensors import safe_open
except ImportError:
    torch = None

VOCAB = ["<pad>", "<unk>", "<bos>", "你", "好", "世", "界"]
CONFIG = {"vocab": len(VOCAB), "n_layer": 1, "n_head": 2, "n_embd": 8,
          "context": 64, "dropout": 0.0}
PROVENANCE = {"license": "test fixture", "data_classification": "public-test",
              "attribution": "Synthetic Chinese paragraphs authored for this test.",
              "sources": ["test_train_from_random_weights"]}


def write_spec(path, config=CONFIG, vocab=VOCAB, provenance=PROVENANCE):
    path.write_text(json.dumps({"config": config, "vocab": vocab, "provenance": provenance}),
                    encoding="utf-8")


class SpecTests(unittest.TestCase):
    def test_learning_rate_tracks_effective_targets_and_caps_warmup(self):
        self.assertEqual(TRAIN.learning_rate(250, 10_000, 64, 0.001), 0.0005)
        self.assertEqual(TRAIN.learning_rate(500, 10_000, 64, 0.001), 0.001)
        self.assertAlmostEqual(TRAIN.learning_rate(5_250, 10_000, 64, 0.001), 0.00055)
        self.assertAlmostEqual(TRAIN.learning_rate(10_000, 10_000, 64, 0.001), 0.0001)
        self.assertEqual(TRAIN.learning_rate(3_200, 1_000_000, 64, 0.001), 0.0005)

    def test_explicit_shape_vocabulary_and_provenance_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix="pretraining-spec-", dir=CACHE) as directory:
            path = Path(directory) / "spec.json"
            for shape in ({"n_layer": 6, "n_head": 4, "n_embd": 256},
                          {"n_layer": 8, "n_head": 8, "n_embd": 448}):
                config = dict(CONFIG, **shape)
                write_spec(path, config=config)
                self.assertEqual(TRAIN.read_spec(path), (config, VOCAB, PROVENANCE))

    def test_rejects_invalid_spec_at_the_boundary(self):
        with tempfile.TemporaryDirectory(prefix="invalid-pretraining-spec-", dir=CACHE) as directory:
            path = Path(directory) / "spec.json"
            for config in (dict(CONFIG, n_head=3), dict(CONFIG, context=128),
                           dict(CONFIG, n_layer=True), dict(CONFIG, n_embd=0),
                           dict(CONFIG, dropout=float("nan")), dict(CONFIG, dropout=1),
                           dict(CONFIG, unexpected="field")):
                with self.subTest(config=config):
                    write_spec(path, config=config)
                    with self.assertRaisesRegex(ValueError, "vocabulary/model configuration"):
                        TRAIN.read_spec(path)
            for vocab in (VOCAB[:-1], VOCAB[:-1] + ["你"], VOCAB[:-1] + ["两个"],
                          VOCAB[:-1] + ["\ud800"], ["<unk>", "<pad>"] + VOCAB[2:]):
                with self.subTest(vocab=vocab):
                    write_spec(path, vocab=vocab)
                    with self.assertRaisesRegex(ValueError, "vocabulary/model configuration"):
                        TRAIN.read_spec(path)
            write_spec(path, provenance={"license": "test fixture"})
            with self.assertRaisesRegex(ValueError, "provenance requires"):
                TRAIN.read_spec(path)


@unittest.skipIf(torch is None, "install training requirements for actual tensor checks")
class TextTrainingTests(unittest.TestCase):
    def test_targets_cover_each_character_once_without_crossing_paragraphs(self):
        with tempfile.TemporaryDirectory(prefix="text-mask-", dir=CACHE) as directory:
            directory = Path(directory)
            source = directory / "text.txt"
            source.write_text("你好世界你\n好\n\n未知\n", encoding="utf-8")
            corpus, counts = TRAIN.encode_text(source, VOCAB, 3, directory / "tokens.u16")
            self.assertEqual(corpus.tolist(), [[2, 3, 4, 5], [5, 6, 3, 0], [2, 4, 0, 0]])
            self.assertEqual(counts["effective_targets"], 6)
            self.assertEqual(counts["padding_targets"], 3)
            self.assertEqual(counts["unknown_targets"], 2)
            self.assertEqual(counts["source_targets"], 8)
            self.assertEqual(counts["masked_unknown_targets"], 0)
            self.assertEqual(counts["skipped_no_target_blocks"], 1)
            inputs, targets, count = TRAIN.text_batch(corpus, slice(None), "cpu", torch)
            self.assertEqual(inputs.shape, (3, 3))
            self.assertEqual(count, 6)
            model = TRAIN.initialize_random(CONFIG, torch, 7)
            model.eval()
            logits, loss = model(inputs, targets)
            expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)
            self.assertEqual(loss.item(), expected.item())
            _, capped, count = TRAIN.text_batch(corpus, slice(None), "cpu", torch, token_limit=4)
            self.assertEqual(count, 4)
            self.assertEqual(capped.tolist(), [[3, 4, 5], [6, 0, 0], [0, 0, 0]])

    def test_unknown_targets_are_ignored_without_corrupting_input_context(self):
        with tempfile.TemporaryDirectory(prefix="unknown-targets-", dir=CACHE) as directory:
            directory = Path(directory)
            source = directory / "text.txt"
            source.write_text("你未好\n未未未好\n", encoding="utf-8")
            corpus, counts = TRAIN.encode_text(source, VOCAB, 3, directory / "tokens.u16")
            self.assertEqual(corpus.tolist(), [[2, 3, 1, 4], [1, 4, 0, 0]])
            self.assertEqual(counts["effective_targets"], 3)
            self.assertEqual(counts["unknown_targets"], 4)
            self.assertEqual(counts["masked_unknown_targets"], 1)
            self.assertEqual(counts["skipped_no_target_blocks"], 1)
            inputs, targets, count = TRAIN.text_batch(corpus, slice(0, 1), "cpu", torch)
            self.assertEqual(inputs.tolist(), [[2, 3, 1]])
            self.assertEqual(targets.tolist(), [[3, 0, 4]])
            self.assertEqual(count, 2)
            inputs, targets, count = TRAIN.text_batch(corpus, slice(0, 1), "cpu", torch, token_limit=1)
            self.assertEqual(inputs.tolist(), [[2, 3, 1]])
            self.assertEqual(targets.tolist(), [[3, 0, 0]])
            self.assertEqual(count, 1)
            inputs, targets, count = TRAIN.text_batch(corpus, slice(1, 2), "cpu", torch)
            self.assertEqual(inputs.tolist(), [[1]])
            self.assertEqual(targets.tolist(), [[4]])
            self.assertEqual(count, 1)
            source.write_text("未知\n\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no effective training targets"):
                TRAIN.encode_text(source, VOCAB, 3, directory / "empty.u16")

    def test_length_batches_cover_all_blocks_with_bounded_positions_and_equal_loss(self):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="length-batches-", dir=CACHE) as directory:
            directory = Path(directory)
            source = directory / "text.txt"
            source.write_text("你\n" * 70 + "你未好\n" * 23 + "你好世界" * 33 + "\n",
                              encoding="utf-8")
            corpus, counts = TRAIN.encode_text(source, VOCAB, 64, directory / "tokens.u16")
            groups = TRAIN.length_groups(corpus)
            batches = TRAIN.text_batches(groups, 64, np.random.default_rng(7))
            self.assertEqual(sorted(np.concatenate(batches).tolist()), list(range(len(corpus))))
            self.assertEqual([tuple(batch) for batch in batches],
                             [tuple(batch) for batch in TRAIN.text_batches(groups, 64, np.random.default_rng(7))])
            self.assertNotEqual([tuple(batch) for batch in batches],
                                [tuple(batch) for batch in TRAIN.text_batches(groups, 64, np.random.default_rng(8))])
            model = TRAIN.initialize_random(CONFIG, torch, 7)
            model.eval()
            full = torch.tensor(np.array(corpus, dtype="int64"))
            full_targets = full[:, 1:].clone()
            full_targets[full_targets == 1] = 0
            total = count = 0
            underfilled = []
            with torch.inference_mode():
                _, full_loss = model(full[:, :-1], full_targets)
                for indices in batches:
                    inputs, targets, n = TRAIN.text_batch(corpus, indices, "cpu", torch)
                    spans = np.count_nonzero(corpus[indices, 1:], axis=1)
                    self.assertTrue(np.all(spans == inputs.shape[1]))
                    self.assertLessEqual(inputs.numel(), 64)
                    if len(indices) < 64 // inputs.shape[1]:
                        underfilled.append(inputs.shape[1])
                    _, loss = model(inputs, targets)
                    total += loss.item() * n
                    count += n
            self.assertEqual(len(underfilled), len(set(underfilled)))
            self.assertEqual(count, counts["effective_targets"])
            self.assertTrue(math.isclose(total / count, full_loss.item(), rel_tol=1e-6))

    def test_random_weights_use_explicit_provenance_and_known_target_budget(self):
        with tempfile.TemporaryDirectory(prefix="fresh-pretraining-", dir=CACHE) as directory:
            directory = Path(directory)
            source = directory / "spec.json"
            write_spec(source)
            train, dev = directory / "train.txt", directory / "dev.txt"
            train.write_text("你未好世界你好\n", encoding="utf-8")
            dev.write_text("世未界你\n", encoding="utf-8")
            output = directory / "output"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/pretrain_model.py"),
                "--train", str(train), "--dev", str(dev), "--spec", str(source),
                "--output-dir", str(output), "--device", "cpu", "--precision", "fp32",
                "--epochs", "1", "--max-tokens", "5", "--batch", "2", "--threads", "1"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(selection["training_effective_targets"], 5)
            self.assertEqual(selection["optimizer_steps"], 1)
            self.assertEqual(selection["stop_reason"], "token_limit")
            recipe = json.loads((output / "recipe.json").read_text())
            self.assertEqual(recipe["counts"]["train"]["effective_targets"], 6)
            self.assertEqual(recipe["counts"]["train"]["unknown_targets"], 1)
            self.assertEqual(recipe["counts"]["dev"]["effective_targets"], 3)
            self.assertEqual(recipe["provenance"], PROVENANCE)
            self.assertEqual(recipe["spec"]["sha256"], TRAIN.digest(source))
            self.assertEqual(recipe["data"], {"train": TRAIN.digest(train), "dev": TRAIN.digest(dev)})
            self.assertEqual(recipe["batching"]["position_budget"], 128)
            self.assertEqual(recipe["optimizer"]["planned_effective_targets"], 5)
            self.assertEqual(recipe["optimizer"]["warmup_effective_targets"], 0.25)
            with safe_open(str(output / "best-f32.safetensors"), framework="pt") as artifact:
                metadata = artifact.metadata()
                self.assertEqual(metadata["initialization"], "random")
                self.assertEqual(metadata["data_classification"], "public-test")
                self.assertEqual(metadata["license"], PROVENANCE["license"])
                self.assertEqual(metadata["attribution"], PROVENANCE["attribution"])
                self.assertEqual(json.loads(metadata["provenance"]), PROVENANCE)
                self.assertEqual(metadata["training_spec_sha256"], TRAIN.digest(source))
                self.assertNotIn("parent_model_sha256", metadata)
                self.assertEqual(json.loads(metadata["config"]), CONFIG)
                tok = artifact.get_tensor("tok.weight")
                self.assertEqual(tuple(tok.shape), (len(VOCAB), 8))
                self.assertTrue(torch.isfinite(tok).all())
            config, _, _ = TRAIN.read_spec(source)
            a = TRAIN.initialize_random(config, torch, 42)
            b = TRAIN.initialize_random(config, torch, 42)
            c = TRAIN.initialize_random(config, torch, 43)
            self.assertTrue(torch.equal(a.tok.weight, b.tok.weight))
            self.assertFalse(torch.equal(a.tok.weight, c.tok.weight))

    def test_requested_device_precision_does_not_silently_fallback(self):
        with self.assertRaisesRegex(ValueError, "CPU training requires"):
            TRAIN.check_runtime("cpu", "bf16", torch)
        with mock.patch.object(torch.xpu, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no device fallback"):
                TRAIN.check_runtime("xpu", "bf16", torch)

    def test_bf16_selects_math_attention_and_restores_the_previous_policy(self):
        def enabled():
            backend = torch.backends.cuda
            return (backend.math_sdp_enabled(), backend.flash_sdp_enabled(),
                    backend.mem_efficient_sdp_enabled(), backend.cudnn_sdp_enabled())

        previous = enabled()
        with mock.patch.object(torch, "autocast", return_value=nullcontext()) as amp:
            with TRAIN.autocast("xpu", "bf16", torch):
                self.assertEqual(enabled(), (True, False, False, False))
            amp.assert_called_once_with(device_type="xpu", dtype=torch.bfloat16)
        self.assertEqual(enabled(), previous)
        with TRAIN.autocast("cpu", "fp32", torch):
            self.assertEqual(enabled(), previous)


if __name__ == "__main__":
    unittest.main()
