"""Check the supervised ranking contract without a large training run."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_ranker", ROOT / "scripts/train_ranker.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)

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
        with tempfile.TemporaryDirectory(prefix="train-ranker-data-") as directory:
            path = Path(directory) / "cases.jsonl"
            for value in [[], None, 7]:
                path.write_text(json.dumps(value) + "\n")
                with self.assertRaises(ValueError) as error:
                    TRAIN.read_cases(path)
                self.assertIn(f"{path}:1:", str(error.exception))
                self.assertIn("JSON object", str(error.exception))

    def test_case_ids_and_target_fields_match_cli_input(self):
        with tempfile.TemporaryDirectory(prefix="train-ranker-fields-") as directory:
            path = Path(directory) / "cases.jsonl"
            base = {"input": "ni", "gold": "你"}
            path.write_text(json.dumps(base) + "\n" + json.dumps(dict(base, id="")) + "\n")
            self.assertEqual([case["id"] for case in TRAIN.read_cases(path)], ["1", "2"])
            for fields in ({"id": None}, {"id": 0}, {"id": []}, {"golds": ["你"]}):
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
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, 8)
        self.assertEqual([group["penalties"] for group in groups], [[0.0, 3.0], [0.0, 0.0]])
        self.assertEqual(statuses, {"trainable": 2, "neutral_language_group": 1,
                                   "gold_outside_shortlist": 1})
        self.assertEqual(sum(statuses.values()), len(cases))
        cases[0]["golds"] = ["现", "西安"]
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, 8)
        self.assertEqual(statuses["no_negative_candidate"], 1)
        with self.assertRaisesRegex(ValueError, "denominator"):
            TRAIN.training_groups(cases, {"rows": rows[:-1]}, 8)

    def test_equal_input_or_gold_cannot_cross_training_development(self):
        train = [{"input": "xian", "golds": ["西安"]}]
        for dev in [[{"input": "xian", "golds": ["先"]}],
                    [{"input": "xian1", "golds": ["西安"]}]]:
            with self.assertRaisesRegex(ValueError, "train/dev overlap"):
                TRAIN.reject_overlap(train, dev)
        TRAIN.reject_overlap(train, [{"input": "nihao", "golds": ["你好"]}])

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
        groups = [{"context": "前文" * 20, "texts": ["西安", "先"]}]
        tokens, mask, bounds = TRAIN.encode(groups, vocab, 4, torch)
        self.assertEqual(tokens.tolist(), [[2, 4, 5, 6], [2, 4, 7, 0]])
        self.assertEqual(mask.tolist(), [[0, 1, 1], [0, 1, 0]])
        self.assertEqual(bounds, [(0, 2)])
        tokens, mask, _ = TRAIN.encode(groups, vocab, 3, torch)
        self.assertEqual(tokens.tolist(), [[2, 5, 6], [2, 7, 0]])
        self.assertEqual(mask.tolist(), [[1, 1], [1, 0]])

    def test_loss_contains_static_prior_and_leader_promotion_margin(self):
        values = torch.tensor([-2.0, -1.0], requires_grad=True)
        groups = [{"penalties": [0.0, 2.0], "gold_indices": [1]}]
        actual = TRAIN.ranking_loss(values, groups, [(0, 2)], torch)
        expected = torch.nn.functional.cross_entropy(torch.tensor([[-1.75, -3.0]]), torch.tensor([1]))
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)
        actual.backward()
        self.assertGreater(values.grad[0].item(), 0)
        self.assertLess(values.grad[1].item(), 0)

    def test_export_preserves_metadata_without_optional_training_fields(self):
        model = torch.nn.Linear(2, 2)
        metadata = {"format": "chinese-ime-lm", "license": "GPL-3.0-or-later",
                    "attribution": "public fixture"}
        with tempfile.TemporaryDirectory(prefix="train-ranker-test-") as directory:
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
