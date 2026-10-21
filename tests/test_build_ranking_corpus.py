"""Public text sampling, pronunciation alignment and inherited split boundaries."""
from collections import Counter
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/training-tests"
CACHE.mkdir(parents=True, exist_ok=True)
SPEC = importlib.util.spec_from_file_location("build_ranking_corpus", ROOT / "scripts/build_ranking_corpus.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def row(group="a", split="train", source="lccc", paragraph=0, text=None):
    return {"source": source, "group": group, "paragraph": paragraph, "split": split,
            "text": text or "今天我们一起到公园散步然后回家喝茶聊天讨论明天的工作安排"}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RankingCorpusTests(unittest.TestCase):
    def test_annotation_uses_whole_han_run_before_slicing_polyphonic_target(self):
        paragraphs = [row(group="polyphone", text="银行")]
        with mock.patch.object(BUILD, "lazy_pinyin", wraps=BUILD.lazy_pinyin) as annotate:
            proposals = BUILD.document_proposals(paragraphs, Counter())
            annotate.assert_not_called()
            cases = [BUILD.annotate(proposal) for proposal in proposals]
        self.assertTrue(annotate.call_args_list)
        self.assertTrue(all(call.args[0] == "银行" for call in annotate.call_args_list))
        single = next(case for case in cases if case["gold"] == "行")
        self.assertEqual((single["input"], single["context"]), ("hang", "银"))
        self.assertEqual(single["source_group"], "lccc:polyphone")
        self.assertEqual(single["group_id"], single["source_group"])

    def test_hash_sampling_caps_documents_and_context_without_looking_at_candidates(self):
        text = "今天我们一起到公园散步然后回家喝茶聊天讨论明天的工作安排" * 4
        context_counts = Counter()
        length_counts = Counter()
        for i in range(80):
            paragraphs = [row(group=str(i), paragraph=j, text=text) for j in range(5)]
            counts = Counter()
            proposals = BUILD.document_proposals(paragraphs, counts)
            self.assertEqual(proposals, BUILD.document_proposals(paragraphs, Counter()))
            cases = [BUILD.annotate(proposal) for proposal in proposals]
            self.assertEqual(len(cases), 4)
            self.assertEqual(counts["proposed_slots"], counts["fitting_proposals"])
            self.assertEqual(counts["fitting_proposals"], counts["document_slot_not_selected"] + counts["selected_document_slots"])
            for case in cases:
                self.assertRegex(case["input"], r"^[a-z]+$")
                self.assertTrue(BUILD.HAN.fullmatch(case["gold"]))
                self.assertLessEqual(len(case["context"]), 32)
                if case["context"]:
                    self.assertIn(case["context"] + case["gold"], text)
                context_counts[bool(case["context"])] += 1
                length_counts[len(case["gold"])] += 1
            for split in ("dev", "holdout"):
                heldout = BUILD.document_proposals([dict(value, split=split) for value in paragraphs], Counter())
                self.assertEqual(len(heldout), 1)
        self.assertTrue(0.15 < context_counts[False] / sum(context_counts.values()) < 0.35)
        self.assertEqual(length_counts[1], 80)
        self.assertEqual(length_counts[2], 80)
        self.assertEqual(sum(length_counts[n] for n in (3, 4)), 80)
        self.assertEqual(sum(length_counts[n] for n in range(5, 9)), 80)

    def test_unavailable_pronunciation_is_counted_without_replacing_it(self):
        with tempfile.TemporaryDirectory(prefix="ranking-pronunciation-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            write_rows(documents, [row()])
            with mock.patch.object(BUILD, "lazy_pinyin", side_effect=lambda text, **kwargs: ["?" for _ in text]):
                recipe = BUILD.build(documents, directory / "out", directory / "evidence", [])
            counts = recipe["splits"]["train"]["source_stats"]["lccc"]
            self.assertEqual(counts["non_ascii_full_pinyin"], counts["annotation_pool_cases"])
            self.assertEqual(counts["written"], 0)

    def test_evaluation_exclusion_uses_complete_query_and_expands_gold_sets(self):
        with tempfile.TemporaryDirectory(prefix="ranking-exclusions-", dir=CACHE) as temporary:
            path = Path(temporary) / "eval.jsonl"
            write_rows(path, [{"input": "ru", "context": "出", "golds": ["入", "如"]},
                              {"id": "annotation-without-query", "normalized_text": "出现"}])
            excluded, files = BUILD.evaluation_queries([path])
            self.assertEqual(len(excluded), 2)
            self.assertIn(BUILD.query_key({"input": "ru", "context": "出", "gold": "入"}), excluded)
            self.assertNotIn(BUILD.query_key({"input": "ru", "context": "加", "gold": "入"}), excluded)
            self.assertEqual(files[0]["sha256"], BUILD.digest(path))

    def test_cli_uses_public_evaluations_and_explicit_repeatable_exclusions(self):
        with tempfile.TemporaryDirectory(prefix="ranking-cli-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            write_rows(documents, [row()])
            public = directory / "data/eval/public.jsonl"
            public.parent.mkdir(parents=True)
            public_case = BUILD.annotate(BUILD.document_proposals([row()], Counter())[0])
            write_rows(public, [public_case])
            args = ["build_ranking_corpus.py", "--documents", str(documents),
                    "--output-dir", str(directory / "baseline"), "--evidence-dir", str(directory / "baseline-evidence")]
            with mock.patch.object(BUILD, "ROOT", directory), mock.patch("sys.argv", args), mock.patch("builtins.print"):
                BUILD.main()
            recipe = json.loads((directory / "baseline-evidence/ranking-data-recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(recipe["exclusion_files"], [{"path": str(public.resolve()), "sha256": BUILD.digest(public)}])
            self.assertFalse((directory / "results").exists())
            baseline = read_rows(directory / "baseline/ranking-train.jsonl")
            self.assertNotIn(BUILD.query_key(public_case), {BUILD.query_key(case) for case in baseline})
            extra_paths = [directory / f"extra-{index}.jsonl" for index in range(2)]
            for path, case in zip(extra_paths, baseline[:2]):
                write_rows(path, [case])
            args = ["build_ranking_corpus.py", "--documents", str(documents),
                    "--output-dir", str(directory / "out"), "--evidence-dir", str(directory / "evidence"),
                    "--exclude", str(extra_paths[0]), "--exclude", str(extra_paths[1])]
            with mock.patch.object(BUILD, "ROOT", directory), mock.patch("sys.argv", args), mock.patch("builtins.print"):
                BUILD.main()
            cases = read_rows(directory / "out/ranking-train.jsonl")
            self.assertEqual({BUILD.query_key(case) for case in cases},
                             {BUILD.query_key(case) for case in baseline[2:]})
            recipe = json.loads((directory / "evidence/ranking-data-recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(recipe["exclusion_files"],
                             [{"path": str(path.resolve()), "sha256": BUILD.digest(path)} for path in [public, *extra_paths]])

    def test_cli_preserves_missing_explicit_exclusion_error(self):
        with tempfile.TemporaryDirectory(prefix="ranking-missing-exclusion-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            write_rows(documents, [row()])
            missing = directory / "missing.jsonl"
            args = ["build_ranking_corpus.py", "--documents", str(documents),
                    "--output-dir", str(directory / "out"), "--evidence-dir", str(directory / "evidence"),
                    "--exclude", str(missing)]
            with mock.patch.object(BUILD, "ROOT", directory), mock.patch("sys.argv", args):
                with self.assertRaises(FileNotFoundError) as error:
                    BUILD.main()
            self.assertEqual(error.exception.filename, str(missing))
            self.assertFalse((directory / "out").exists())
            self.assertFalse((directory / "evidence").exists())

    def test_invalid_document_boundaries_report_file_and_line(self):
        with tempfile.TemporaryDirectory(prefix="ranking-boundaries-", dir=CACHE) as temporary:
            path = Path(temporary) / "documents.jsonl"
            for rows in ([row(), row(split="dev", paragraph=1)],
                         [row(), row()], [row(), row(group="b"), row(paragraph=1)],
                         [dict(row(), paragraph=True)], [dict(row(), source=[])],
                         [dict(row(), split="validation")]):
                with self.subTest(rows=rows):
                    write_rows(path, rows)
                    with self.assertRaises(ValueError) as error:
                        list(BUILD.read_documents(path))
                    self.assertIn(str(path) + ":", str(error.exception))

    def test_build_preserves_splits_hashes_counts_limits_and_exact_query_dedup(self):
        with tempfile.TemporaryDirectory(prefix="ranking-build-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents, exclusions = directory / "documents.jsonl", directory / "eval.jsonl"
            rows = []
            ownership = {}
            for split in BUILD.SPLITS:
                for source in ("lccc", "wiki"):
                    for i in range(30):
                        group = f"{split}-{i}"
                        rows += [row(group, split, source, paragraph=j) for j in range(2)]
                        ownership[source + ":" + group] = split
            write_rows(documents, rows)
            limits = {source: 2 for source in BUILD.TRAIN_LIMITS}
            BUILD.build(documents, directory / "baseline", directory / "baseline-evidence", [], limits, limits)
            blocked = read_rows(directory / "baseline" / "ranking-train.jsonl")[0]
            write_rows(exclusions, [blocked])
            first = BUILD.build(documents, directory / "out-a", directory / "evidence-a", [exclusions], limits, limits)
            second = BUILD.build(documents, directory / "out-b", directory / "evidence-b", [exclusions], limits, limits)
            self.assertEqual(first["pypinyin_version"], version("pypinyin"))
            self.assertEqual(first["documents"]["sha256"], BUILD.digest(documents))
            queries, ids = set(), set()
            excluded_count = duplicate_count = budget_count = 0
            for split in BUILD.SPLITS:
                summary = first["splits"][split]
                path = directory / "out-a" / f"ranking-{split}.jsonl"
                cases = read_rows(path)
                self.assertEqual(summary["sha256"], BUILD.digest(path))
                self.assertEqual(summary["sha256"], second["splits"][split]["sha256"])
                self.assertEqual(summary["cases"], len(cases))
                self.assertEqual(sum(summary["source"].values()), len(cases))
                self.assertEqual(sum(summary["target_length"].values()), len(cases))
                self.assertEqual(sum(summary["context"].values()), len(cases))
                by_document = Counter()
                for case in cases:
                    self.assertEqual(ownership[case["source_group"]], split)
                    self.assertEqual(case["group_id"], case["source_group"])
                    self.assertNotIn(case["id"], ids)
                    self.assertNotIn(BUILD.query_key(case), queries)
                    self.assertNotEqual(BUILD.query_key(case), BUILD.query_key(blocked))
                    ids.add(case["id"])
                    queries.add(BUILD.query_key(case))
                    by_document[case["source_group"]] += 1
                self.assertTrue(all(count <= (4 if split == "train" else 1) for count in by_document.values()))
                for source, stats in summary["source_stats"].items():
                    self.assertLessEqual(stats["written"], 2)
                    self.assertLessEqual(stats["pypinyin_calls"], 3)
                    self.assertEqual(stats["pypinyin_calls"], stats["annotation_pool_cases"])
                    self.assertEqual(stats["sampled_document_cases"], stats["hash_pool_not_selected"] + stats["annotation_pool_cases"])
                    self.assertEqual(stats["annotation_pool_cases"], sum(stats[key] for key in
                        ("non_ascii_full_pinyin", "existing_evaluation_query", "duplicate_exact_query", "unique_eligible_cases")))
                    self.assertEqual(stats["unique_eligible_cases"] - stats["written"], stats["source_budget_not_selected"])
                    excluded_count += stats.get("existing_evaluation_query", 0)
                    duplicate_count += stats.get("duplicate_exact_query", 0)
                    budget_count += stats["source_budget_not_selected"]
            self.assertGreater(excluded_count, 0)
            self.assertGreater(duplicate_count, 0)
            self.assertGreater(budget_count, 0)
            self.assertEqual(first["splits"]["train"]["source_stats"]["cqia"]["written"], 0)
            with self.assertRaises(FileExistsError):
                BUILD.build(documents, directory / "out-a", directory / "evidence-a", [exclusions], limits, limits)


if __name__ == "__main__":
    unittest.main()
