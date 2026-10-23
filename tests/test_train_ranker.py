"""Check the supervised ranking contract without a large training run."""
import copy
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


def scored_row(identity, candidates, indices=None, dictionary=None, language=None, rank=2):
    if indices is None:
        indices = list(range(len(candidates)))
    if dictionary is None:
        dictionary = [0.0] * len(indices)
    if language is None:
        language = [-2.0] * len(indices)
    scores = [{"index": index, "dictionary": prior, "language": logp}
              for index, prior, logp in zip(indices, dictionary, language, strict=True)]
    return {"id": identity, "rank": rank, "candidates": candidates,
            "scoring": {"candidates": candidates,
                        "promotion": {"scores": scores, "index": None, "margin": 0.25}}}


class TrainingDataTests(unittest.TestCase):
    def test_only_actual_source_top1_rank_groups_use_retention(self):
        cases = [{"id": str(i), "input": "xian", "context": "前文", "golds": ["西安"]}
                 for i in range(3)]
        rows = [scored_row("0", [candidate("现"), candidate("西安")], rank=1),
                scored_row("1", [candidate("现"), candidate("西安")], rank=2),
                scored_row("2", [candidate("现"), candidate("先")], rank=None)]
        rows[0]["candidates"] = list(reversed(rows[0]["candidates"]))
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, VOCAB, 64)
        self.assertEqual(statuses, {"trainable": 2, "gold_outside_scoring": 1})
        self.assertIs(groups[0]["retain_top1"], True)
        self.assertEqual(groups[0]["gold_indices"], [1])
        self.assertNotIn("retain_top1", groups[1])
        self.assertTrue(all(group["composed_leader"] for group in groups[:2]))
        self.assertIs(groups[2]["target_only"], True)
        self.assertNotIn("retain_top1", groups[2])

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

    def test_training_uses_actual_scoring_candidates_indices_and_dictionary(self):
        cases = [{"id": "actual", "input": "xian", "context": "前文", "golds": ["西安"]}]
        candidates = [candidate("现", 20.0, "ExactDictionary"), candidate("先", -30.0),
                      candidate("不参与"), candidate("西安", 19.0)]
        row = scored_row("actual", candidates, [0, 3, 1], dictionary=[0.0, -3.5, -0.75])
        row["candidates"] = [candidate("最终首选")]
        groups, statuses = TRAIN.training_groups(cases, {"rows": [row]}, VOCAB, 64)
        self.assertEqual(statuses, {"trainable": 1})
        self.assertEqual(groups, [{"id": "actual", "context": "前文", "texts": ["现", "西安", "先"],
                                  "source_indices": [0, 3, 1], "known_mask": [True] * 3,
                                  "composed_leader": False,
                                  "penalties": [0.0, 3.5, 0.75], "margin": 0.25, "gold_indices": [1]}])

    def test_training_keeps_unscored_neutral_missing_and_all_gold_cases_in_statuses(self):
        cases = [{"id": str(i), "input": "xian", "context": "前文", "golds": ["西安"]}
                 for i in range(4)]
        rows = [scored_row("0", [candidate("现"), candidate("西安")], []),
                scored_row("1", [candidate("现"), candidate("西安")], language=[0.0, 0.0]),
                scored_row("2", [candidate("现"), candidate("先")]),
                scored_row("3", [candidate("西安"), candidate("先")])]
        cases[0]["golds"] = ["入"]
        cases[1]["golds"] = ["入"]
        cases[3]["golds"] = ["西安", "先"]
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, VOCAB, 64)
        self.assertEqual(groups, [{"id": "2", "context": "前文", "texts": ["西安"],
                                  "known_mask": [True], "target_only": True}])
        self.assertEqual(statuses, {"protected_or_no_competition": 1, "neutral_language_group": 1,
                                   "gold_outside_scoring": 1, "no_negative_candidate": 1})
        self.assertEqual(sum(statuses.values()), len(cases))
        with self.assertRaisesRegex(ValueError, "denominator"):
            TRAIN.training_groups(cases, {"rows": rows[:-1]}, VOCAB, 64)
        rows[0]["scoring"] = None
        with self.assertRaisesRegex(ValueError, "source model scoring"):
            TRAIN.training_groups(cases, {"rows": rows}, VOCAB, 64)

    def test_missing_target_supervision_keeps_actual_menu_and_source_indices_unchanged(self):
        case = {"id": "missing", "input": "xian", "context": "前文",
                "golds": ["西安", "入", "乳", "西安" * 32]}
        row = scored_row("missing", [candidate("现"), candidate("前文")])
        original = json.loads(json.dumps(row))
        groups, statuses = TRAIN.training_groups([case], {"rows": [row]}, VOCAB, 64)
        self.assertEqual(statuses, {"gold_outside_scoring": 1})
        self.assertEqual(groups, [{"id": "missing", "context": "前文", "texts": ["西安", "入"],
                                  "known_mask": [True, True], "target_only": True}])
        self.assertEqual(row, original)
        self.assertNotIn("source_indices", groups[0])
        self.assertNotIn("penalties", groups[0])
        case["golds"] = ["乳"]
        groups, statuses = TRAIN.training_groups([case], {"rows": [row]}, VOCAB, 64)
        self.assertEqual(groups, [])
        self.assertEqual(statuses, {"gold_outside_scoring": 1})

    def test_invalid_cli_scoring_indices_and_contributions_are_rejected(self):
        cases = [{"id": "1", "input": "xian", "context": "前文", "golds": ["西安"]}]
        candidates = [candidate("现"), candidate("西安")]
        for indices in [[0, 2], [0, -1], [0, True], [0, 0], [1, 0]]:
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "scoring indices"):
                    TRAIN.training_groups(cases, {"rows": [scored_row("1", candidates, indices)]}, VOCAB, 64)
        for field, value in [("dictionary", 0.1), ("dictionary", float("nan")),
                             ("language", float("inf"))]:
            with self.subTest(field=field, value=value):
                row = scored_row("1", candidates)
                row["scoring"]["promotion"]["scores"][1][field] = value
                with self.assertRaisesRegex(ValueError, "scoring contribution"):
                    TRAIN.training_groups(cases, {"rows": [row]}, VOCAB, 64)
        for damaged in [candidate("西安", consumed=3), candidate("西安", source="Completion")]:
            with self.subTest(damaged=damaged):
                with self.assertRaisesRegex(ValueError, "complete conversion"):
                    TRAIN.training_groups(cases, {"rows": [scored_row("1", [candidates[0], damaged])]}, VOCAB, 64)

    def test_composed_golds_use_the_same_training_groups_and_gold_sets(self):
        cases = [{"id": str(i), "input": "xian", "context": "前文", "golds": golds}
                 for i, golds in enumerate([["西安"], ["西安", "先"], ["现"], ["西安"]])]
        candidates = [candidate("现", source="ExactDictionary"), candidate("西安"),
                      candidate("先", source="ExactDictionary")]
        rows = [scored_row(case["id"], candidates) for case in cases]
        rows[-1] = scored_row("3", [candidate("现"), candidate("西安"), candidate("先")])
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, VOCAB, 64)
        self.assertEqual(statuses, {"trainable": 4})
        self.assertEqual(sum(statuses.values()), len(cases))
        self.assertEqual([group["id"] for group in groups], ["0", "1", "2", "3"])
        for group in groups:
            self.assertEqual(group["texts"], ["现", "西安", "先"])
            self.assertEqual(group["source_indices"], [0, 1, 2])
            self.assertEqual(group["known_mask"], [True, True, True])
        self.assertEqual([group["gold_indices"] for group in groups], [[1], [1, 2], [0], [1]])

    def test_oov_policy_keeps_actual_scored_candidates_and_gold_identity_with_explicit_exclusions(self):
        cases = [{"id": str(i), "input": "xian", "context": "前文", "golds": golds}
                 for i, golds in enumerate([["入"], ["入"], ["如"], ["乳"], ["西安"], ["乳", "入"]])]
        rows = [scored_row("0", [candidate("如"), candidate("乳"), candidate("入")]),
                scored_row("1", [candidate("乳"), candidate("如"), candidate("入")]),
                scored_row("2", [candidate("如"), candidate("乳")]),
                scored_row("3", [candidate("如"), candidate("乳"), candidate("入")]),
                scored_row("4", [candidate("现" * 8), candidate("西安")], language=[0.0, 0.0]),
                scored_row("5", [candidate("如"), candidate("乳"), candidate("入")])]
        groups, statuses = TRAIN.training_groups(cases, {"rows": rows}, VOCAB, 64)
        self.assertEqual(statuses, {"trainable": 2, "neutral_language_group_oov_leader": 1,
                                   "neutral_language_group_insufficient_known_candidates": 1,
                                   "neutral_language_group": 1, "gold_only_unrepresented": 1})
        self.assertEqual(sum(statuses.values()), len(cases))
        self.assertEqual([group["id"] for group in groups], ["0", "5"])
        for group in groups:
            self.assertEqual(group["texts"], ["如", "乳", "入"])
            self.assertEqual(group["source_indices"], [0, 1, 2])
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

    def test_roundtrip_compares_behavior_and_scoring_identity_without_diagnostic_float_noise(self):
        row = scored_row("1", [candidate("现"), candidate("西安")])
        row.update(rank=2, static_rank=2)
        before = report([row])
        after = json.loads(json.dumps(before))
        after["rows"][0]["scoring"]["promotion"]["scores"][1].update(
            language=-2.0000002, dictionary=-0.0000001, total=-2.0000003)
        self.assertEqual(TRAIN.cli_behavior(before), TRAIN.cli_behavior(after))
        mutations = [lambda row: row.update(rank=1), lambda row: row.update(static_rank=1),
                     lambda row: row["candidates"][0].update(score_bits=1),
                     lambda row: row["scoring"]["candidates"][1].update(consumed=3),
                     lambda row: row["scoring"]["promotion"]["scores"][1].update(index=0),
                     lambda row: row["scoring"]["promotion"].update(index=1)]
        for change in mutations:
            damaged = json.loads(json.dumps(before))
            change(damaged["rows"][0])
            self.assertNotEqual(TRAIN.cli_behavior(before), TRAIN.cli_behavior(damaged))

    def test_group_selection_keeps_missing_cases_and_checks_candidate_identity(self):
        cases = [{"id": "1", "input": "xian", "context": "去", "golds": ["西安"]},
                 {"id": "2", "input": "xian", "context": "时间", "golds": ["先"]},
                 {"id": "3", "input": "nihao", "context": "", "golds": ["你好"]}]
        rows = [{"id": "1", "rank": 1, "static_rank": 1, "candidates": [candidate("西安")]},
                {"id": "2", "rank": None, "static_rank": None, "candidates": []},
                {"id": "3", "rank": 1, "static_rank": 1, "candidates": [candidate("你好")]}]
        before = report(rows)
        result = TRAIN.compare_reports(cases, before, before, before)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["group_top1_fraction"], [3, 4])
        self.assertEqual(result["multi_target_groups"], 1)
        self.assertEqual(result["all_targets_top1_groups"], 0)
        self.assertTrue(result["candidate_identity_unchanged"])
        self.assertTrue(result["static_candidate_identity_preserved"])
        self.assertEqual(result["added_candidates"], 0)
        changed = json.loads(json.dumps(before))
        changed["rows"][0]["candidates"][0]["score_bits"] += 1
        with self.assertRaisesRegex(ValueError, "candidate identity"):
            TRAIN.compare_reports(cases, before, changed, before)

    def test_selection_accepts_full_composed_recall_and_replaces_only_matching_special_candidates(self):
        cases = [{"id": "1", "input": "xian", "golds": ["仙"]}]
        exact = candidate("现", source="ExactDictionary")
        before = report([{"id": "1", "rank": None, "static_rank": None,
                          "candidates": [exact, candidate("西安", source="Completion"),
                                         candidate("先", source="MissingLetter { at: 1, to: 105, ends: [4] }")]}])
        after = report([{"id": "1", "rank": 1, "static_rank": None,
                         "candidates": [candidate("仙"), candidate("西安"), exact, candidate("先")]}])
        result = TRAIN.compare_reports(cases, before, after, before)
        self.assertFalse(result["candidate_identity_unchanged"])
        self.assertTrue(result["static_candidate_identity_preserved"])
        self.assertEqual(result["added_candidates"], 3)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["rescues"], [{"id": "1", "before_rank": None, "after_rank": 1}])

    def test_selection_allows_weight_dependent_beams_and_pairs_initial_model_ranks(self):
        cases = [{"id": "1", "input": "xian", "golds": ["先"]}]
        exact = candidate("现", source="ExactDictionary")
        static = report([{"id": "1", "rank": None, "static_rank": None, "candidates": [exact]}])
        before = report([{"id": "1", "rank": 1, "static_rank": None,
                          "candidates": [candidate("先"), exact]}])
        after = report([{"id": "1", "rank": None, "static_rank": None,
                         "candidates": [candidate("仙"), exact]}])
        result = TRAIN.compare_reports(cases, before, after, static)
        self.assertFalse(result["candidate_identity_unchanged"])
        self.assertEqual(result["added_candidates"], 1)
        self.assertEqual(result["breakages"], [{"id": "1", "before_rank": 1, "after_rank": None}])
        before = json.loads(json.dumps(before))
        before["rows"][0]["candidates"][1]["score_bits"] += 1
        with self.assertRaisesRegex(ValueError, "candidate identity"):
            TRAIN.compare_reports(cases, before, after, static)

    def test_selection_rejects_damaged_old_identity_and_unsupported_recall(self):
        cases = [{"id": "1", "input": "xian", "golds": ["现"]}]
        exact = candidate("现", source="ExactDictionary")
        before = report([{"id": "1", "rank": 1, "static_rank": 1, "candidates": [exact]}])
        for field, value in [("score_bits", exact["score_bits"] + 1),
                             ("source", "Composed"), ("consumed", 3)]:
            with self.subTest(field=field):
                changed = json.loads(json.dumps(before))
                changed["rows"][0]["candidates"][0][field] = value
                with self.assertRaisesRegex(ValueError, "candidate identity"):
                    TRAIN.compare_reports(cases, before, changed, before)
        for added in [candidate("先", source="ExactDictionary"), candidate("先", consumed=3),
                      candidate("现", score=-1.0)]:
            with self.subTest(added=added):
                changed = json.loads(json.dumps(before))
                changed["rows"][0]["candidates"].append(added)
                with self.assertRaisesRegex(ValueError, "candidate identity"):
                    TRAIN.compare_reports(cases, before, changed, before)


@unittest.skipIf(torch is None, "install scripts/training-requirements.txt for tensor checks")
class TrainingTensorTests(unittest.TestCase):
    def test_safe_retention_has_zero_gradient_and_stays_in_group_mean(self):
        values = torch.tensor([-1., -2., -3., -1.5, -1., -1.1], requires_grad=True)
        groups = [{"composed_leader": False, "gold_indices": [gold], "known_mask": [True] * 3,
                   "penalties": [0.] * 3, "margin": 0.25, "retain_top1": True}
                  for gold in [0, 1]]
        losses = TRAIN.training_losses(values, groups, [(0, 3), (3, 6)], torch)
        self.assertEqual(losses[0].item(), 0)
        self.assertAlmostEqual(losses[1].item(), 0.15, places=6)
        losses.mean().backward()
        torch.testing.assert_close(values.grad, torch.tensor([0., 0., 0., 0., -0.5, 0.5]), rtol=0, atol=0)

    def test_retention_uses_best_accepted_gold_and_hardest_non_gold(self):
        group = {"composed_leader": False, "gold_indices": [1, 2], "known_mask": [True] * 5,
                 "penalties": [0.] * 5, "margin": 0.25, "retain_top1": True}
        for hardest in [0, 3, 4]:
            with self.subTest(hardest=hardest):
                values = torch.tensor([-3., -1.5, -1.6, -3., -3.], requires_grad=True)
                with torch.no_grad():
                    values[hardest] = -1.4
                loss = TRAIN.training_losses(values, [group], [(0, 5)], torch)[0]
                expected = 0.6 if hardest == 0 else 0.35
                self.assertAlmostEqual(loss.item(), expected, places=6)
                loss.backward()
                gradient = torch.zeros(5)
                gradient[1], gradient[hardest] = -1, 1
                torch.testing.assert_close(values.grad, gradient, rtol=0, atol=0)
                if hardest != 0:
                    summed_gold_loss = torch.relu(0.25 + values[hardest] - values[[1, 2]].logsumexp(0))
                    self.assertEqual(summed_gold_loss.item(), 0)

    def test_retention_keeps_dictionary_leader_margin_and_oov_tie(self):
        group = {"composed_leader": False, "gold_indices": [1], "known_mask": [True, True],
                 "penalties": [0., 0.5], "margin": 0.25, "retain_top1": True}
        for gold_value, expected in [(-0.1, 0.1), (0., 0.)]:
            with self.subTest(gold_value=gold_value):
                values = torch.tensor([-1., gold_value], requires_grad=True)
                loss = TRAIN.training_losses(values, [group], [(0, 2)], torch)[0]
                self.assertAlmostEqual(loss.item(), expected, places=6)
        values = torch.tensor([-2., 99., -1.8], requires_grad=True)
        group.update(gold_indices=[2], known_mask=[True, False, True], penalties=[0., 0., 0.])
        loss = TRAIN.training_losses(values, [group], [(0, 3)], torch)[0]
        self.assertAlmostEqual(loss.item(), 0.3, places=6)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([1., 0., -1.]), rtol=0, atol=0)

    def test_rank_loss_uses_common_leader_length_before_oov_tie(self):
        group = {"composed_leader": True, "texts": ["现", "西安", "㨎㨎㨎"], "known_mask": [True, True, False],
                 "penalties": [0., 0., 0.], "margin": 0.25, "gold_indices": [0]}
        direct = dict(group, composed_leader=False)
        direct_values = torch.tensor([-2., -1.2, 99.], requires_grad=True)
        direct_loss = TRAIN.training_losses(direct_values, [direct], [(0, 3)], torch)[0]
        direct_scores = torch.tensor([-1.75, -1.2, -2.])
        self.assertAlmostEqual(direct_loss.item(), torch.nn.functional.cross_entropy(
            direct_scores[None], torch.tensor([0])).item(), places=6)
        values = torch.tensor([-2., -1.2, 99.], requires_grad=True)
        loss = TRAIN.training_losses(values, [group], [(0, 3)], torch)[0]
        scores = torch.tensor([-1.75, -2.4, -2.])
        expected = torch.nn.functional.cross_entropy(scores[None], torch.tensor([0]))
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        probabilities = scores.softmax(0)
        expected_gradient = torch.tensor([probabilities[0] + probabilities[2] - 1,
                                          2 * probabilities[1], 0.])
        torch.testing.assert_close(values.grad, expected_gradient, rtol=1e-6, atol=1e-7)

        group["retain_top1"] = True
        values = torch.tensor([-2., -0.8, 99.], requires_grad=True)
        loss = TRAIN.training_losses(values, [group], [(0, 3)], torch)[0]
        self.assertAlmostEqual(loss.item(), 0.4, places=6)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([-1., 2., 0.]), rtol=0, atol=0)

    def test_text_kl_uses_teacher_direction_full_vocabulary_and_effective_token_mean(self):
        teacher_logits = torch.tensor([
            [[3., -1., 0., 2., -2.], [1., 2., 3., 4., 5.], [-1., 0., 2., 1., 3.]],
            [[9., 8., 7., 6., 5.], [0., 1., -2., 4., 2.], [5., 4., 3., 2., 1.]]])
        student_logits = (teacher_logits + torch.tensor([1.5, -0.4, 0.2, -1., 0.7])).requires_grad_()
        targets = torch.tensor([[3, 0, 4], [0, 4, 0]])
        valid = targets.ne(0)
        teacher_prob = teacher_logits[valid].softmax(-1)
        student_logp = student_logits[valid].log_softmax(-1)
        expected = (teacher_prob * (teacher_prob.log() - student_logp)).sum() / valid.sum()
        reverse = (student_logp.exp() * (student_logp - teacher_prob.log())).sum() / valid.sum()
        loss = TRAIN.text_teacher_kl(student_logits, teacher_logits, targets, torch)
        torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
        self.assertGreater(abs(loss.item() - reverse.item()), 0.001)
        loss.backward()
        expected_gradient = torch.zeros_like(student_logits)
        expected_gradient[valid] = (student_logp.detach().exp() - teacher_prob) / valid.sum()
        torch.testing.assert_close(student_logits.grad, expected_gradient, rtol=1e-6, atol=1e-7)
        self.assertGreater(student_logits.grad[..., :3].abs().sum().item(), 0)
        changed_student = student_logits.detach().clone().requires_grad_()
        changed_teacher = teacher_logits.clone()
        with torch.no_grad():
            changed_student[~valid] = float("nan")
            changed_teacher[~valid] = float("inf")
        changed_loss = TRAIN.text_teacher_kl(changed_student, changed_teacher, targets, torch)
        changed_loss.backward()
        torch.testing.assert_close(changed_loss, loss, rtol=0, atol=0)
        torch.testing.assert_close(changed_student.grad, student_logits.grad, rtol=0, atol=0)

    def test_training_entry_keeps_teacher_frozen_and_student_dropout_rng(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.path.insert(0, str(ROOT / "vendor/chinese-ime-training/neural"))
        from model import CharLM, Config
        import initialize
        torch.manual_seed(79)
        cfg = Config(vocab=len(VOCAB), n_layer=1, n_head=2, n_embd=16, context=8, dropout=0.3)
        student = CharLM(cfg)
        baseline = copy.deepcopy(student)
        initial_weights = {name: value.clone() for name, value in student.state_dict().items()}
        teachers, text_outputs = [], []

        def clone_teacher(source):
            before = torch.get_rng_state().clone()
            teacher = copy.deepcopy(source)
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            original_forward = teacher.forward

            def forward(inputs):
                self.assertFalse(teacher.training)
                self.assertFalse(torch.is_grad_enabled())
                before = torch.get_rng_state().clone()
                output = original_forward(inputs)
                self.assertTrue(torch.equal(torch.get_rng_state(), before))
                return output

            teacher.forward = forward
            teachers.append(teacher)
            return teacher

        def student_forward(module, inputs, output):
            if module is student:
                self.assertTrue(student.training)
                self.assertEqual(len(inputs), 1)
                text_outputs.append((inputs[0].clone(), output[0].detach().clone(), torch.get_rng_state().clone()))

        with tempfile.TemporaryDirectory(prefix="train-ranker-kl-", dir=TEST_CACHE) as directory:
            directory = Path(directory)
            train_path, dev_path = directory / "train.jsonl", directory / "dev.jsonl"
            train_path.write_text(json.dumps({"id": "train", "input": "xian", "context": "前文", "gold": "西安"}) + "\n")
            dev_path.write_text(json.dumps({"id": "dev", "input": "ru", "gold": "入"}) + "\n")
            text_path = directory / "text.txt"
            text_path.write_text("前乳西安\n")
            source_row = scored_row("train", [candidate("现"), candidate("西安")], rank=1)
            source_row["candidates"] = list(reversed(source_row["candidates"]))
            dev_report = report([{"id": "dev", "rank": 1, "static_rank": 1,
                                  "candidates": [candidate("入", consumed=2)]}])

            def evaluate(binary, cases, output, model=None):
                return (report([source_row]) if cases == train_path else dev_report), []

            output_dir = directory / "output"
            argv = ["train_ranker.py", "--model", str(directory / "parent.safetensors"),
                    "--binary", str(directory / "yanliu"), "--train", str(train_path),
                    "--dev", str(dev_path), "--text-train", str(text_path),
                    "--output-dir", str(output_dir), "--epochs", "1", "--batch", "1", "--seed", "83"]
            with mock.patch.object(initialize, "load_initial_model", return_value=(student, cfg, VOCAB, {})), \
                    mock.patch.object(TRAIN, "copy") as cloning, \
                    mock.patch.object(TRAIN, "digest", return_value="fixture"), \
                    mock.patch.object(TRAIN, "cli_evaluate", side_effect=evaluate), \
                    student.register_forward_hook(student_forward), \
                    mock.patch.object(torch, "set_num_interop_threads"), \
                    mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                cloning.deepcopy.side_effect = clone_teacher
                TRAIN.main()

            self.assertEqual(len(teachers), 1)
            self.assertEqual(len(text_outputs), 1)
            self.assertEqual(text_outputs[0][0].tolist(), [[2, VOCAB.index("前"), 1, VOCAB.index("西")]])
            teacher = teachers[0]
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                for parameter in teacher.parameters()))
            self.assertIs(teacher.head.weight, teacher.tok.weight)
            self.assertNotEqual(teacher.tok.weight.data_ptr(), student.tok.weight.data_ptr())
            for name, value in teacher.state_dict().items():
                torch.testing.assert_close(value, initial_weights[name], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, initial_weights[name])
                                for name, value in student.state_dict().items()))
            training = json.loads((output_dir / "training.json").read_text())
            self.assertGreater(training[0]["text_teacher_kl"], 0)
            self.assertEqual(training[0]["text_effective_targets"], 3)
            groups, _ = TRAIN.training_groups(TRAIN.read_cases(train_path), report([source_row]), VOCAB, cfg.context)
            tokens, mask, _ = TRAIN.encode(groups, VOCAB, cfg.context, torch)
            baseline.train()
            torch.manual_seed(83)
            TRAIN.language_scores(baseline, tokens, mask, torch)
            expected_logits, _ = baseline(text_outputs[0][0])
            torch.testing.assert_close(text_outputs[0][1], expected_logits, rtol=0, atol=0)
            self.assertTrue(torch.equal(text_outputs[0][2], torch.get_rng_state()))

    def test_target_only_loss_uses_gold_suffix_mask_and_equal_accepted_target_gradients(self):
        groups = [{"context": "前文", "texts": ["西安", "入"],
                   "known_mask": [True, True], "target_only": True}]
        tokens, mask, bounds = TRAIN.encode(groups, VOCAB, 8, torch)
        self.assertEqual(mask.tolist(), [[0, 0, 1, 1], [0, 0, 1, 0]])
        self.assertEqual(bounds, [(0, 2)])
        values = torch.tensor([-3.0, -1.0], requires_grad=True)
        loss = TRAIN.training_losses(values, groups, bounds, torch).mean()
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([-0.5, -0.5]))

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
            {"composed_leader": True, "context": "前文西安先", "texts": ["现", "西安", "先前文"],
             "known_mask": [True, True, True], "penalties": [0.0, 0.5, 0.0], "margin": 0.25,
             "gold_indices": [2]},
            {"composed_leader": True, "context": "", "texts": ["如", "乳", "入"],
             "known_mask": [True, False, True], "penalties": [0.0, 0.25, 0.0], "margin": 0.25,
             "gold_indices": [1, 2]},
            {"composed_leader": True, "context": "前文" * 8, "texts": ["如", "入入"],
             "known_mask": [True, True], "penalties": [0.0, 0.5], "margin": 0.25,
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
                expected_loss = TRAIN.training_losses(expected_scores, groups, bounds, torch).mean()
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
                loss = TRAIN.training_losses(scores, groups, bounds, torch).mean()
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
        groups = [{"composed_leader": False, "margin": 0.25, "penalties": [0.0, 2.0],
                   "gold_indices": [1], "known_mask": [True, True]}]
        actual = TRAIN.training_losses(values, groups, [(0, 2)], torch).mean()
        expected = torch.nn.functional.cross_entropy(torch.tensor([[-1.75, -3.0]]), torch.tensor([1]))
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)
        actual.backward()
        self.assertGreater(values.grad[0].item(), 0)
        self.assertLess(values.grad[1].item(), 0)

    def test_oov_language_score_uses_the_leaders_value_and_gradient(self):
        values = torch.tensor([-2.0, 99.0, -1.0], requires_grad=True)
        groups = [{"composed_leader": False, "margin": 0.25, "penalties": [0.0, 1.0, 0.0], "gold_indices": [2],
                   "known_mask": [True, False, True],}]
        loss = TRAIN.training_losses(values, groups, [(0, 3)], torch).mean()
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
        loss = TRAIN.training_losses(values, groups, [(0, 3)], torch).mean()
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
