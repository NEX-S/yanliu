"""Check the supervised ranking contract without a large training run."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TEST_CACHE = ROOT / ".cache/training-tests"
TEST_CACHE.mkdir(parents=True, exist_ok=True)
SPEC = importlib.util.spec_from_file_location("train_ranker", ROOT / "scripts/train_ranker.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)
VOCAB = ["<pad>", "<unk>", "<bos>"] + list("现西安先前文如入")

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    torch = None


def candidate(text, score=0.0, source="Composed", consumed=4):
    return {"text": text, "consumed": consumed, "source": source,
            "score_bits": struct.unpack("<I", struct.pack("<f", score))[0]}


def report(rows):
    return {"rows": rows, "buckets": {"all": {"n": len(rows),
            "top1": sum(row["rank"] == 1 for row in rows)}}}


class TrainingDataTests(unittest.TestCase):
    def test_nonobject_case_reports_input_path_and_line(self):
        with tempfile.TemporaryDirectory(prefix="train-ranker-data-", dir=TEST_CACHE) as directory:
            path = Path(directory) / "cases.jsonl"
            for value in [[], None, 7]:
                path.write_text(json.dumps(value) + "\n")
                with self.assertRaises(ValueError) as error:
                    TRAIN.read_cases(path)
                self.assertIn(f"{path}:1:", str(error.exception))
                self.assertIn("JSON object", str(error.exception))

    def test_case_ids_and_target_fields_match_cli_input(self):
        with tempfile.TemporaryDirectory(prefix="train-ranker-fields-", dir=TEST_CACHE) as directory:
            path = Path(directory) / "cases.jsonl"
            base = {"input": "ni", "gold": "你"}
            path.write_text(json.dumps(base) + "\n" + json.dumps(dict(base, id="")) + "\n")
            self.assertEqual([case["id"] for case in TRAIN.read_cases(path)], ["1", "2"])
            for fields in ({"id": None}, {"id": 0}, {"id": []}, {"golds": ["你"]},
                           {"source_group": ""}, {"source_group": []}):
                with self.subTest(fields=fields):
                    path.write_text(json.dumps(dict(base, **fields)) + "\n")
                    with self.assertRaisesRegex(ValueError, f"{path}:1:"):
                        TRAIN.read_cases(path)
            path.write_text(json.dumps(base) + "\n{broken JSON}\n")
            with self.assertRaises(ValueError) as error:
                TRAIN.read_cases(path)
            self.assertIn(f"{path}:2:", str(error.exception))
            self.assertIsInstance(error.exception.__cause__, json.JSONDecodeError)

    def test_shortlist_protects_empty_context_and_full_consumption(self):
        case = {"input": "xian", "context": ""}
        exact = [candidate("现", source="ExactDictionary"), candidate("西安")]
        self.assertEqual(TRAIN.shortlist(case, exact), [])
        case["context"] = "我去"
        self.assertEqual(TRAIN.shortlist(case, exact), [0, 1])
        exact[0]["source"] = "User"
        self.assertEqual(TRAIN.shortlist(case, exact), [])
        exact[0]["source"] = "Composed"
        exact += [candidate("先", consumed=2), candidate("西岸", source="Completion")]
        self.assertEqual(TRAIN.shortlist(case, exact), [0, 1])
        exact[0]["consumed"] = 2
        self.assertEqual(TRAIN.shortlist(case, exact), [])

    def test_shortlist_reserves_different_lengths_then_fills_to_nine(self):
        case = {"input": "xian", "context": "前文"}
        candidates = [candidate(chr(0x4e00 + i)) for i in range(12)]
        candidates[10]["text"] = "两个"
        self.assertEqual(TRAIN.shortlist(case, candidates), [0, 1, 2, 3, 4, 5, 6, 10, 7])

    def test_training_uses_bits_prior_and_excludes_neutral_or_missing_targets(self):
        cases = [{"id": str(i), "input": "xian", "context": "前文", "golds": ["西安"]}
                 for i in range(4)]
        rows = [{"id": "0", "candidates": [candidate("现", 2.0, "ExactDictionary"),
                                               candidate("西安", 0.5)]},
                {"id": "1", "candidates": [candidate("现", 2.0), candidate("西安", 0.5)]},
                {"id": "2", "candidates": [candidate("B站"), candidate("西安")]},
                {"id": "3", "candidates": [candidate("现"), candidate("先")]}]
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, 8, VOCAB)
        self.assertEqual([group["penalties"] for group in groups], [[0.0, 3.0], [0.0, 0.0]])
        self.assertEqual([group["known_mask"] for group in groups], [[True, True], [True, True]])
        self.assertEqual(statuses, {"trainable": 2, "neutral_language_group": 1,
                                   "gold_outside_shortlist": 1})
        self.assertEqual(sum(statuses.values()), len(cases))
        cases[0]["golds"] = ["现", "西安"]
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, 8, VOCAB)
        self.assertEqual(statuses["no_negative_candidate"], 1)
        with self.assertRaisesRegex(ValueError, "denominator"):
            TRAIN.training_groups(cases, {"rows": rows[:-1]}, 8, VOCAB)

    def test_oov_policy_keeps_shortlist_and_gold_identity_with_explicit_exclusions(self):
        cases = [{"id": str(i), "input": "xian", "context": "前文", "golds": golds}
                 for i, golds in enumerate([["入"], ["入"], ["如"], ["乳"], ["西安"], ["乳", "入"]])]
        rows = [{"id": "0", "candidates": [candidate("如"), candidate("乳"), candidate("入")]},
                {"id": "1", "candidates": [candidate("乳"), candidate("如"), candidate("入")]},
                {"id": "2", "candidates": [candidate("如"), candidate("乳")]},
                {"id": "3", "candidates": [candidate("如"), candidate("乳"), candidate("入")]},
                {"id": "4", "candidates": [candidate("现" * 8), candidate("西安")]},
                {"id": "5", "candidates": [candidate("如"), candidate("乳"), candidate("入")]}]
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, 8, VOCAB)
        self.assertEqual(statuses, {"trainable": 2, "neutral_language_group_oov_leader": 1,
                                   "neutral_language_group_insufficient_known_candidates": 1,
                                   "neutral_language_group": 1, "gold_only_unrepresented": 1})
        self.assertEqual(sum(statuses.values()), len(cases))
        self.assertEqual([group["id"] for group in groups], ["0", "5"])
        for group in groups:
            self.assertEqual(group["texts"], ["如", "乳", "入"])
            self.assertEqual(group["static_indices"], [0, 1, 2])
            self.assertEqual(group["known_mask"], [True, False, True])
        self.assertEqual(groups[0]["gold_indices"], [2])
        self.assertEqual(groups[1]["gold_indices"], [1, 2])

    def test_common_words_in_different_contexts_or_source_groups_are_allowed(self):
        train = [{"input": "xian", "context": "我去", "golds": ["西安"], "source_group": "a"}]
        TRAIN.reject_overlap(train, [{"input": "xian", "context": "很早", "golds": ["先"],
                                      "source_group": "b"}])
        TRAIN.reject_overlap(train, [{"input": "xian", "context": "旅游目的地", "golds": ["西安"],
                                      "source_group": "b"}])

    def test_source_group_and_normalized_substantive_duplicates_are_rejected(self):
        train = [{"input": "xian", "context": "我去", "golds": ["西安"], "source_group": "a"}]
        with self.assertRaisesRegex(ValueError, "1 source_group"):
            TRAIN.reject_overlap(train, [{"input": "nihao", "context": "", "golds": ["你好"],
                                          "source_group": "a"}])
        with self.assertRaisesRegex(ValueError, "1 duplicate"):
            TRAIN.reject_overlap(train, [{"input": "ＸＩＡＮ", "context": "我 去！", "golds": ["先", "西安"],
                                          "source_group": "b"}])

    def test_group_selection_keeps_missing_cases_and_checks_candidate_identity(self):
        cases = [{"id": "1", "input": "xian", "context": "去", "golds": ["西安"]},
                 {"id": "2", "input": "xian", "context": "时间", "golds": ["先"]},
                 {"id": "3", "input": "nihao", "context": "", "golds": ["你好"]}]
        rows = [{"id": "1", "rank": 1, "static_rank": 1, "candidates": [candidate("西安")]},
                {"id": "2", "rank": None, "static_rank": None, "candidates": []},
                {"id": "3", "rank": 1, "static_rank": 1, "candidates": [candidate("你好")]}]
        before = report(rows)
        result = TRAIN.compare_reports(cases, before, before)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["group_top1_fraction"], [3, 4])
        self.assertEqual(result["multi_target_groups"], 1)
        self.assertEqual(result["all_targets_top1_groups"], 0)
        changed = json.loads(json.dumps(before))
        changed["rows"][0]["candidates"][0]["score_bits"] += 1
        with self.assertRaisesRegex(ValueError, "candidate identity"):
            TRAIN.compare_reports(cases, before, changed)


@unittest.skipIf(torch is None, "install scripts/training-requirements.txt for tensor checks")
class TrainingTensorTests(unittest.TestCase):
    def test_encoding_leaves_bos_room_using_actual_window_and_longest_candidate(self):
        vocab = ["<pad>", "<unk>", "<bos>", "前", "文", "西", "安", "先"]
        groups = [{"context": "前文" * 20, "texts": ["西安", "先"], "known_mask": [True, True]}]
        tokens, mask, bounds = TRAIN.encode(groups, vocab, 4, torch)
        self.assertEqual(tokens.tolist(), [[2, 4, 5, 6], [2, 4, 7, 0]])
        self.assertEqual(mask.tolist(), [[0, 1, 1], [0, 1, 0]])
        self.assertEqual(bounds, [(0, 2)])
        tokens, mask, _ = TRAIN.encode(groups, vocab, 3, torch)
        self.assertEqual(tokens.tolist(), [[2, 5, 6], [2, 7, 0]])
        self.assertEqual(mask.tolist(), [[1, 1], [1, 0]])

    def test_oov_placeholder_does_not_shorten_known_candidates_context(self):
        vocab = ["<pad>", "<unk>", "<bos>", "前", "文", "西", "安", "先"]
        groups = [{"context": "前文" * 20, "texts": ["西安", "未知文字", "先"],
                   "known_mask": [True, False, True]}]
        tokens, mask, bounds = TRAIN.encode(groups, vocab, 5, torch)
        self.assertEqual(tokens.tolist(), [[2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 7, 0]])
        self.assertEqual(mask.tolist(), [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0]])
        self.assertEqual(bounds, [(0, 3)])

    def test_candidate_only_projection_matches_full_forward_scores_loss_and_gradients(self):
        sys.path.insert(0, str(ROOT / "vendor/chinese-ime-training/neural"))
        from model import CharLM, Config
        groups = [
            {"context": "前文西安先", "texts": ["现", "西安", "先前文"],
             "known_mask": [True, True, True], "penalties": [0.0, 0.5, 0.0],
             "gold_indices": [1]},
            {"context": "", "texts": ["如", "乳", "入"],
             "known_mask": [True, False, True], "penalties": [0.0, 0.25, 0.0],
             "gold_indices": [1, 2]},
            {"context": "前文" * 8, "texts": ["如", "入入"],
             "known_mask": [True, True], "penalties": [0.0, 0.5],
             "gold_indices": [1]},
        ]
        original_groups = json.loads(json.dumps(groups))
        tokens, mask, bounds = TRAIN.encode(groups, VOCAB, 8, torch)
        original_tokens, original_mask = tokens.clone(), mask.clone()
        self.assertEqual(tokens.shape, (8, 8))
        self.assertEqual(int(mask.sum()), 12)
        self.assertLess(int(mask.sum()), mask.numel())
        torch.manual_seed(71)
        model = CharLM(Config(vocab=len(VOCAB), n_layer=2, n_head=2, n_embd=16,
                              context=8, dropout=0.3))
        for training in [False, True]:
            with self.subTest(training=training):
                model.train(training)
                model.zero_grad(set_to_none=True)
                torch.manual_seed(73)
                logits, _ = model(tokens[:, :-1])
                logp = logits.float().log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
                expected_scores = (logp * mask).sum(-1) / mask.sum(-1)
                expected_loss = TRAIN.ranking_loss(expected_scores, groups, bounds, torch)
                expected_rng = torch.get_rng_state()
                expected_loss.backward()
                expected_gradients = {name: parameter.grad.clone()
                                      for name, parameter in model.named_parameters()}

                model.zero_grad(set_to_none=True)
                torch.manual_seed(73)
                with mock.patch.object(model.head, "forward", wraps=model.head.forward) as head:
                    scores = TRAIN.language_scores(model, tokens, mask, torch)
                self.assertEqual(head.call_count, 1)
                self.assertEqual(head.call_args.args[0].shape, (int(mask.sum()), model.cfg.n_embd))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                loss = TRAIN.ranking_loss(scores, groups, bounds, torch)
                torch.testing.assert_close(scores, expected_scores, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)
                loss.backward()
                for name, parameter in model.named_parameters():
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(parameter.grad, expected_gradients[name],
                                                   rtol=1e-5, atol=1e-7)
                self.assertEqual(groups, original_groups)
                self.assertTrue(torch.equal(tokens, original_tokens))
                self.assertTrue(torch.equal(mask, original_mask))
                self.assertEqual(bounds, [(0, 3), (3, 6), (6, 8)])

    def test_loss_contains_static_prior_and_leader_promotion_margin(self):
        values = torch.tensor([-2.0, -1.0], requires_grad=True)
        groups = [{"penalties": [0.0, 2.0], "gold_indices": [1], "known_mask": [True, True]}]
        actual = TRAIN.ranking_loss(values, groups, [(0, 2)], torch)
        expected = torch.nn.functional.cross_entropy(torch.tensor([[-1.75, -3.0]]), torch.tensor([1]))
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)
        actual.backward()
        self.assertGreater(values.grad[0].item(), 0)
        self.assertLess(values.grad[1].item(), 0)

    def test_oov_language_score_uses_the_leaders_value_and_gradient(self):
        values = torch.tensor([-2.0, 99.0, -1.0], requires_grad=True)
        groups = [{"penalties": [0.0, 1.0, 0.0], "gold_indices": [2],
                   "known_mask": [True, False, True]}]
        loss = TRAIN.ranking_loss(values, groups, [(0, 3)], torch)
        scores = torch.tensor([-1.75, -3.0, -1.0])
        expected = torch.nn.functional.cross_entropy(scores[None], torch.tensor([2]))
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        probabilities = scores.softmax(0)
        self.assertAlmostEqual(values.grad[0].item(), (probabilities[0] + probabilities[1]).item(), places=6)
        self.assertEqual(values.grad[1].item(), 0.0)
        self.assertAlmostEqual(values.grad[2].item(), (probabilities[2] - 1).item(), places=6)

        groups[0]["gold_indices"] = [1, 2]
        values = torch.tensor([-2.0, -99.0, -1.0], requires_grad=True)
        loss = TRAIN.ranking_loss(values, groups, [(0, 3)], torch)
        expected = scores.logsumexp(0) - scores[[1, 2]].logsumexp(0)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        self.assertEqual(values.grad[1].item(), 0.0)
        self.assertGreater(values.grad[0].item(), 0.0)
        self.assertLess(values.grad[2].item(), 0.0)

    def test_export_preserves_metadata_without_optional_training_fields(self):
        model = torch.nn.Linear(2, 2)
        metadata = {"format": "chinese-ime-lm", "license": "GPL-3.0-or-later",
                    "attribution": "public fixture"}
        with tempfile.TemporaryDirectory(prefix="train-ranker-test-", dir=TEST_CACHE) as directory:
            artifacts = TRAIN.export_model(model, metadata, "parent", "recipe", Path(directory),
                                           0, 0, torch, save_file)
            for precision in ["f32", "int8"]:
                with safe_open(artifacts[precision]["path"], framework="pt") as artifact:
                    actual = artifact.metadata()
                    for key, value in metadata.items():
                        self.assertEqual(actual[key], value)
                    self.assertEqual(actual["additional_optimizer_steps"], "0")
                    self.assertEqual(actual["parent_model_sha256"], "parent")
                    self.assertEqual(actual["training_recipe_sha256"], "recipe")
                    self.assertNotIn("training_steps", actual)
            self.assertEqual(metadata, {"format": "chinese-ime-lm", "license": "GPL-3.0-or-later",
                                        "attribution": "public fixture"})


if __name__ == "__main__":
    unittest.main()
