"""Public text sampling, pronunciation alignment and inherited split boundaries."""
from collections import Counter
import importlib.util
from importlib.metadata import version
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/training-tests"
CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("build_ranking_corpus", ROOT / "scripts/build_ranking_corpus.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def row(group="a", split="train", source="lccc", paragraph=0, text=None):
    return {"source": source, "group": group, "split_group": group, "paragraph": paragraph, "split": split,
            "text": text or "今天我们一起到公园散步然后回家喝茶聊天讨论明天的工作安排"}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RankingCorpusTests(unittest.TestCase):
    def test_fresh_cli_loads_supplement_and_aligns_whole_han_runs(self):
        phrases_readings = {
            "差旅": ["chai", "lv"], "重构": ["chong", "gou"],
            "重命名": ["chong", "ming", "ming"], "代码行": ["dai", "ma", "hang"],
            "去重": ["qu", "chong"], "弹窗": ["tan", "chuang"],
            "乐声": ["yue", "sheng"], "认真地": ["ren", "zhen", "de"],
            "轻轻地": ["qing", "qing", "de"], "系安全带": ["ji", "an", "quan", "dai"],
            "银行": ["yin", "hang"],
            "银行家": ["yin", "hang", "jia"], "音乐节": ["yin", "yue", "jie"],
            "长江流域": ["chang", "jiang", "liu", "yu"], "睡一觉": ["shui", "yi", "jiao"],
        }
        readings = {
            **phrases_readings,
            "他正在认真地工作": ["ta", "zheng", "zai", "ren", "zhen", "de", "gong", "zuo"],
            "必须系安全带": ["bi", "xu", "ji", "an", "quan", "dai"],
            "他向女王鞠躬": ["ta", "xiang", "nv", "wang", "ju", "gong"],
        }
        with tempfile.TemporaryDirectory(prefix="ranking-fresh-pronunciation-", dir=CACHE) as temporary:
            directory = Path(temporary)
            scripts = directory / "scripts"
            scripts.mkdir()
            for name in ("build_ranking_corpus.py", "pinyin_phrases.py"):
                (scripts / name).write_bytes((ROOT / "scripts" / name).read_bytes())
            phrases = directory / BUILD.PINYIN_PHRASES
            phrases.parent.mkdir(parents=True)
            phrases.write_bytes((ROOT / BUILD.PINYIN_PHRASES).read_bytes())
            documents = directory / "documents.jsonl"
            runs = {f"pronunciation-{index}-{repeat}": text
                    for index, text in enumerate(readings) for repeat in range(32)}
            write_rows(documents, [row(group=group, text=text) for group, text in runs.items()])
            output, evidence = directory / "out", directory / "evidence"
            result = subprocess.run([sys.executable, str(scripts / "build_ranking_corpus.py"),
                                     "--documents", str(documents), "--output-dir", str(output),
                                     "--evidence-dir", str(evidence)],
                                    check=True, capture_output=True, text=True, cwd=directory)
            cases = read_rows(output / "ranking-train.jsonl")
            self.assertEqual(json.loads(result.stdout)["train"], len(cases))
            for case in cases:
                text = runs[case["document_group"].split(":", 1)[1]]
                start = len(case["context"]) if case["context"] else text.index(case["gold"])
                self.assertEqual(text[start:start + len(case["gold"])], case["gold"])
                self.assertEqual(case["input"], "".join(readings[text][start:start + len(case["gold"])]), case)
                if case["context"]:
                    self.assertEqual(case["context"], text[:start])
                else:
                    self.assertEqual(case["gold"], text)
            full_readings = {case["gold"]: case["input"] for case in cases}
            for phrase, syllables in phrases_readings.items():
                self.assertEqual(full_readings[phrase], "".join(syllables))
            recipe = json.loads((evidence / "ranking-data-recipe.json").read_text(encoding="utf-8"))
            for key, path in (("pinyin_phrases", phrases), ("pinyin_phrases_loader", scripts / "pinyin_phrases.py")):
                self.assertEqual(recipe[key], {"path": str(path.resolve()), "sha256": BUILD.digest(path)})

    def test_build_requires_valid_supplement_before_output(self):
        for value, error in ((None, FileNotFoundError), (b"invalid JSON", json.JSONDecodeError),
                             (b"{}", ValueError)):
            with self.subTest(value=value), tempfile.TemporaryDirectory(prefix="ranking-invalid-phrases-", dir=CACHE) as temporary:
                directory = Path(temporary)
                documents = directory / "documents.jsonl"
                write_rows(documents, [row()])
                if value is not None:
                    phrases = directory / BUILD.PINYIN_PHRASES
                    phrases.parent.mkdir(parents=True)
                    phrases.write_bytes(value)
                with mock.patch.object(BUILD, "ROOT", directory), mock.patch.object(BUILD, "annotate") as annotate:
                    with self.assertRaises(error):
                        BUILD.build(documents, directory / "out", directory / "evidence", [])
                annotate.assert_not_called()
                self.assertFalse((directory / "out").exists())
                self.assertFalse((directory / "evidence").exists())

    def test_build_rejects_wrong_pypinyin_version_before_output(self):
        with tempfile.TemporaryDirectory(prefix="ranking-pypinyin-version-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            write_rows(documents, [row()])
            with (mock.patch.object(BUILD, "ROOT", directory),
                  mock.patch.object(BUILD.pinyin_phrases, "version", return_value="0.54.0")):
                with self.assertRaisesRegex(ValueError, "pypinyin must be exactly 0.55.0"):
                    BUILD.build(documents, directory / "out", directory / "evidence", [])
            self.assertFalse((directory / "out").exists())
            self.assertFalse((directory / "evidence").exists())

    def test_build_rejects_wrong_jieba_version_before_output(self):
        with tempfile.TemporaryDirectory(prefix="ranking-jieba-version-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            write_rows(documents, [row()])
            actual_version = BUILD.pinyin_phrases.version
            with mock.patch.object(BUILD.pinyin_phrases, "version", side_effect=lambda name:
                                   "0.42.0" if name == "jieba" else actual_version(name)):
                with self.assertRaisesRegex(ValueError, "jieba must be exactly 0.42.1"):
                    BUILD.build(documents, directory / "out", directory / "evidence", [])
            self.assertFalse((directory / "out").exists())
            self.assertFalse((directory / "evidence").exists())

    def test_build_rejects_changed_annotation_inputs_before_recipe(self):
        for changed in ("documents.jsonl", "phrases", "loader", "exclusion.jsonl"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(prefix="ranking-changed-input-", dir=CACHE) as temporary:
                directory = Path(temporary)
                documents, exclusion = directory / "documents.jsonl", directory / "exclusion.jsonl"
                write_rows(documents, [row()])
                exclusion.write_text("", encoding="utf-8")
                phrases = directory / BUILD.PINYIN_PHRASES
                phrases.parent.mkdir(parents=True)
                phrases.write_bytes((ROOT / BUILD.PINYIN_PHRASES).read_bytes())
                loader = directory / "pinyin_phrases.py"
                loader.write_bytes(Path(BUILD.pinyin_phrases.__file__).read_bytes())
                spec = importlib.util.spec_from_file_location("ranking_fixture_phrases", loader)
                helper = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(helper)
                boundary = {"documents.jsonl": documents, "phrases": phrases,
                            "loader": loader, "exclusion.jsonl": exclusion}[changed]
                original = BUILD.annotate

                def annotate(proposal):
                    boundary.write_bytes(boundary.read_bytes() + b"\n")
                    return original(proposal)

                with (mock.patch.object(BUILD, "ROOT", directory), mock.patch.object(BUILD, "pinyin_phrases", helper),
                      mock.patch.object(BUILD, "annotate", side_effect=annotate)):
                    with self.assertRaisesRegex(ValueError, "changed during ranking-case construction"):
                        BUILD.build(documents, directory / "out", directory / "evidence", [exclusion])
                self.assertFalse((directory / "evidence/ranking-data-recipe.json").exists())

    def test_annotation_uses_whole_han_run_before_slicing_polyphonic_target(self):
        paragraphs = [row(group="polyphone", text="银行")]
        with mock.patch.object(BUILD.pinyin_phrases, "syllables", wraps=BUILD.pinyin_phrases.syllables) as annotate:
            proposals = BUILD.document_proposals(paragraphs, Counter())
            annotate.assert_not_called()
            cases = [BUILD.annotate(proposal) for proposal in proposals]
        self.assertTrue(annotate.call_args_list)
        self.assertTrue(all(call.args[0] == "银行" for call in annotate.call_args_list))
        single = next(case for case in cases if case["gold"] == "行")
        self.assertEqual((single["input"], single["context"]), ("hang", "银"))
        self.assertEqual(single["source_group"], "lccc:polyphone")
        self.assertEqual(single["group_id"], single["source_group"])

    def test_context_preserves_previous_dialogue_turn_at_han_run_boundary(self):
        paragraphs = [row(group="previous-turn-2", text="明天去哪里？银行。")]
        proposals = BUILD.document_proposals(paragraphs, Counter())
        bank = next(proposal for proposal in proposals if proposal["case"]["gold"] == "银行")
        self.assertEqual(bank["target_start"], 0)
        self.assertEqual(bank["run"], "银行")
        case = BUILD.annotate(bank)
        self.assertEqual(case["context"], "明天去哪里？")
        self.assertEqual(case["input"], "yinhang")

    def test_contextual_target_can_start_inside_a_word(self):
        proposals = BUILD.document_proposals([row(group="queen-9", text="他向女王鞠躬")], Counter())
        target = next(proposal for proposal in proposals if proposal["case"]["gold"] == "王鞠躬")
        self.assertEqual(target["run"], "他向女王鞠躬")
        self.assertEqual(target["target_start"], 3)
        case = BUILD.annotate(target)
        self.assertEqual((case["context"], case["input"]), ("他向女", "wangjugong"))

    def test_segmentation_prevents_cross_word_de_and_le_misreadings(self):
        for text, start, context, gold, expected in (
                ("对不起您拨打的用户暂时无法接通", 6, "对不起您拨打", "的用户", "deyonghu"),
                ("明天就是周末了", 2, "今天是周五，明天", "就是周末了", "jiushizhoumole")):
            with self.subTest(text=text):
                case = BUILD.annotate({"run": text, "target_start": start,
                                       "case": {"context": context, "gold": gold}})
                self.assertEqual(case["input"], expected)
                self.assertEqual(case["context"], context)

    def test_no_context_uses_complete_naturally_delimited_runs(self):
        for text, targets in (("银行", {"银行"}), ("晴，银行，去公园，明天一起散步。", {"晴", "银行", "去公园", "明天一起散步"}),
                              (" 银行\n", {"银行"}), ("感觉生活好混乱", {"感觉生活好混乱"})):
            with self.subTest(text=text):
                proposals = BUILD.document_proposals([row(group="no-context-36", text=text)], Counter())
                self.assertEqual({proposal["case"]["gold"] for proposal in proposals}, targets)
                for proposal in proposals:
                    self.assertEqual(proposal["target_start"], 0)
                    self.assertEqual(proposal["run"], proposal["case"]["gold"])
                    self.assertEqual(proposal["case"]["context"], "")

    def test_no_context_rejects_digit_and_latin_run_boundaries(self):
        for text in ("1987年落成使用", "年落成使用1987", "GPU计算", "计算CPU", "1987年，使用GPU"):
            with self.subTest(text=text):
                counts = Counter()
                proposals = BUILD.document_proposals([row(group="no-context-36", text=text)], counts)
                self.assertEqual(proposals, [])
                self.assertEqual(counts["no_context_run_not_naturally_delimited"], counts["proposed_slots"])
                self.assertEqual(counts["no_context_run_outside_slot_length"], 0)

    def test_no_context_internal_paragraph_edges_require_real_delimiters(self):
        for texts, targets in ((("他向女", "王鞠躬"), set()),
                               (("你好，", "明天见"), {"你好", "明天见"}),
                               (("你好", "，明天见"), {"你好", "明天见"}),
                               (("你好 ", "明天见"), {"你好", "明天见"}),
                               (("你好", "\n明天见"), {"你好", "明天见"})):
            with self.subTest(texts=texts):
                paragraphs = [row(group="no-context-36", paragraph=index, text=text)
                              for index, text in enumerate(texts)]
                counts = Counter()
                proposals = BUILD.document_proposals(paragraphs, counts)
                self.assertEqual({proposal["case"]["gold"] for proposal in proposals}, targets)
                self.assertTrue(all(proposal["case"]["context"] == "" for proposal in proposals))
                if not targets:
                    self.assertEqual(counts["no_context_run_not_naturally_delimited"], counts["proposed_slots"])

    def test_hash_sampling_caps_documents_and_context_without_looking_at_candidates(self):
        text = "今天我们一起到公园散步然后回家喝茶聊天讨论明天的工作安排" * 4 + "，晴，银行，去公园，明天一起散步。"
        context_counts = Counter()
        length_counts = Counter()
        for i in range(80):
            paragraphs = [row(group=str(i), paragraph=j, text=text) for j in range(5)]
            counts = Counter()
            proposals = BUILD.document_proposals(paragraphs, counts)
            self.assertEqual(proposals, BUILD.document_proposals(paragraphs, Counter()))
            cases = [BUILD.annotate(proposal) for proposal in proposals]
            self.assertEqual(len(cases), 4)
            self.assertEqual(counts["proposed_slots"], sum(counts[name] for name in (
                "fitting_proposals", "run_too_short_for_selected_length_and_context",
                "no_context_run_not_naturally_delimited", "no_context_run_outside_slot_length")))
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

    def test_missing_no_context_length_slots_are_recorded_without_resampling(self):
        with tempfile.TemporaryDirectory(prefix="ranking-missing-length-slots-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            write_rows(documents, [row(group="no-context-36", text="感觉生活好混乱")])
            limits = {source: 5 if source == "lccc" else 0 for source in BUILD.TRAIN_LIMITS}
            recipe = BUILD.build(documents, directory / "out", directory / "evidence", [], train_limits=limits)
            cases = read_rows(directory / "out/ranking-train.jsonl")
            self.assertEqual([(case["gold"], case["context"]) for case in cases], [("感觉生活好混乱", "")])
            self.assertEqual(cases[0]["bucket"], "public-lccc-len5-8-no-context")
            stats = recipe["splits"]["train"]["source_stats"]["lccc"]
            self.assertEqual(stats["proposed_slots"], 4)
            self.assertEqual(stats["no_context_run_outside_slot_length"], 3)
            self.assertEqual(stats["selected_document_slots"], 1)
            self.assertEqual(stats["pypinyin_calls"], 1)
            self.assertEqual(stats["written"], 1)
            self.assertEqual(recipe["rules"]["source_limits"]["train"]["lccc"], 5)

    def test_unavailable_pronunciation_is_counted_without_replacing_it(self):
        with tempfile.TemporaryDirectory(prefix="ranking-pronunciation-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            write_rows(documents, [row()])
            with mock.patch.object(BUILD.pinyin_phrases, "syllables", side_effect=lambda text, **kwargs: ["?" for _ in text]):
                recipe = BUILD.build(documents, directory / "out", directory / "evidence", [])
            counts = recipe["splits"]["train"]["source_stats"]["lccc"]
            self.assertEqual(counts["non_ascii_full_pinyin"], counts["annotation_pool_cases"])
            self.assertEqual(counts["written"], 0)

    def test_evaluation_exclusion_uses_context_and_gold_sets(self):
        with tempfile.TemporaryDirectory(prefix="ranking-exclusions-", dir=CACHE) as temporary:
            path = Path(temporary) / "eval.jsonl"
            write_rows(path, [{"input": "ru", "context": "出", "golds": ["入", "如"]},
                              {"id": "annotation-without-query", "normalized_text": "出现"}])
            excluded, files = BUILD.evaluation_texts([path])
            self.assertEqual(len(excluded), 2)
            self.assertIn(BUILD.text_key({"input": "ru", "context": "出", "gold": "入"}), excluded)
            self.assertNotIn(BUILD.text_key({"input": "ru", "context": "加", "gold": "入"}), excluded)
            self.assertEqual(files[0]["sha256"], BUILD.digest(path))

    def test_text_dedup_ignores_pinyin_and_normalizes_punctuation_and_unicode(self):
        first = {"input": "ＲＵ", "context": "你 来。", "gold": "入"}
        second = {"input": "differentpinyin", "context": "你来！", "gold": "入"}
        self.assertEqual(BUILD.text_key(first), BUILD.text_key(second))
        self.assertNotEqual(BUILD.text_key(first), BUILD.text_key(dict(second, context="他来")))

    def test_changed_pinyin_cannot_bypass_existing_evaluation_text_exclusion(self):
        with tempfile.TemporaryDirectory(prefix="ranking-text-exclusion-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            write_rows(documents, [row(group=f"de-{index}", text="的") for index in range(32)])
            BUILD.build(documents, directory / "baseline", directory / "baseline-evidence", [])
            baseline = read_rows(directory / "baseline/ranking-train.jsonl")
            self.assertEqual([(case["context"], case["gold"], case["input"]) for case in baseline],
                             [("", "的", "de")])
            exclusion = directory / "old-evaluation.jsonl"
            write_rows(exclusion, [dict(baseline[0], input="di")])
            recipe = BUILD.build(documents, directory / "out", directory / "evidence", [exclusion])
            self.assertEqual(read_rows(directory / "out/ranking-train.jsonl"), [])
            self.assertGreater(recipe["splits"]["train"]["source_stats"]["lccc"]["existing_evaluation_text"], 0)

    def test_public_learning_choices_are_excluded_but_queries_without_choices_are_not_labels(self):
        path = ROOT / "data/eval/learning-context.jsonl"
        excluded, _ = BUILD.evaluation_texts([path])
        self.assertEqual(excluded, {
            BUILD.text_key({"input": "xian", "context": "乘火车前往", "gold": "西安"}),
            BUILD.text_key({"input": "xian", "context": "此时此刻", "gold": "现"}),
        })

    def test_cli_uses_public_evaluations_and_explicit_repeatable_exclusions(self):
        with tempfile.TemporaryDirectory(prefix="ranking-cli-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            paragraph = row(text="晴，银行，去公园，明天一起散步。")
            write_rows(documents, [paragraph])
            phrases = directory / BUILD.PINYIN_PHRASES
            phrases.parent.mkdir(parents=True)
            phrases.write_bytes((ROOT / BUILD.PINYIN_PHRASES).read_bytes())
            public = directory / "data/eval/public.jsonl"
            public.parent.mkdir(parents=True)
            public_case = BUILD.annotate(BUILD.document_proposals([paragraph], Counter())[0])
            write_rows(public, [public_case])
            args = ["build_ranking_corpus.py", "--documents", str(documents),
                    "--output-dir", str(directory / "baseline"), "--evidence-dir", str(directory / "baseline-evidence")]
            with mock.patch.object(BUILD, "ROOT", directory), mock.patch("sys.argv", args), mock.patch("builtins.print"):
                BUILD.main()
            recipe = json.loads((directory / "baseline-evidence/ranking-data-recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(recipe["exclusion_files"], [{"path": str(public.resolve()), "sha256": BUILD.digest(public)}])
            self.assertFalse((directory / "results").exists())
            baseline = read_rows(directory / "baseline/ranking-train.jsonl")
            self.assertNotIn(BUILD.text_key(public_case), {BUILD.text_key(case) for case in baseline})
            extra_paths = [directory / f"extra-{index}.jsonl" for index in range(2)]
            for path, case in zip(extra_paths, baseline[:2]):
                write_rows(path, [case])
            args = ["build_ranking_corpus.py", "--documents", str(documents),
                    "--output-dir", str(directory / "out"), "--evidence-dir", str(directory / "evidence"),
                    "--exclude", str(extra_paths[0]), "--exclude", str(extra_paths[1])]
            with mock.patch.object(BUILD, "ROOT", directory), mock.patch("sys.argv", args), mock.patch("builtins.print"):
                BUILD.main()
            cases = read_rows(directory / "out/ranking-train.jsonl")
            self.assertEqual({BUILD.text_key(case) for case in cases},
                             {BUILD.text_key(case) for case in baseline[2:]})
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
                         [dict(row(), split="validation")], [dict(row(), split_group=False)],
                         [row(), dict(row(group="b", split="dev"), split_group="a")]):
                with self.subTest(rows=rows):
                    write_rows(path, rows)
                    with self.assertRaises(ValueError) as error:
                        list(BUILD.read_documents(path))
                    self.assertIn(str(path) + ":", str(error.exception))

    def test_build_preserves_splits_hashes_counts_limits_and_text_dedup(self):
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
            BUILD.build(documents, directory / "baseline", directory / "baseline-evidence", [], limits, limits, limits)
            blocked = read_rows(directory / "baseline" / "ranking-train.jsonl")[0]
            write_rows(exclusions, [blocked])
            first = BUILD.build(documents, directory / "out-a", directory / "evidence-a", [exclusions], limits, limits, limits)
            second = BUILD.build(documents, directory / "out-b", directory / "evidence-b", [exclusions], limits, limits, limits)
            self.assertEqual(first["pypinyin_version"], version("pypinyin"))
            self.assertEqual(first["jieba_version"], version("jieba"))
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
                    self.assertNotIn(BUILD.text_key(case), queries)
                    self.assertNotEqual(BUILD.text_key(case), BUILD.text_key(blocked))
                    ids.add(case["id"])
                    queries.add(BUILD.text_key(case))
                    by_document[case["source_group"]] += 1
                self.assertTrue(all(count <= (4 if split == "train" else 1) for count in by_document.values()))
                for source, stats in summary["source_stats"].items():
                    self.assertLessEqual(stats["written"], 2)
                    self.assertLessEqual(stats["pypinyin_calls"], 3)
                    self.assertEqual(stats["pypinyin_calls"], stats["annotation_pool_cases"])
                    self.assertEqual(stats["sampled_document_cases"], stats["hash_pool_not_selected"] + stats["annotation_pool_cases"])
                    self.assertEqual(stats["annotation_pool_cases"], sum(stats[key] for key in
                        ("non_ascii_full_pinyin", "existing_evaluation_text", "duplicate_text", "unique_eligible_cases")))
                    self.assertEqual(stats["unique_eligible_cases"] - stats["written"], stats["source_budget_not_selected"])
                    excluded_count += stats.get("existing_evaluation_text", 0)
                    duplicate_count += stats.get("duplicate_text", 0)
                    budget_count += stats["source_budget_not_selected"]
            self.assertGreater(excluded_count, 0)
            self.assertGreater(duplicate_count, 0)
            self.assertGreater(budget_count, 0)
            self.assertEqual(first["splits"]["train"]["source_stats"]["cqia"]["written"], 0)
            with self.assertRaises(FileExistsError):
                BUILD.build(documents, directory / "out-a", directory / "evidence-a", [exclusions], limits, limits, limits)

    def test_fresh_holdout_has_independent_budgets_and_keeps_real_split_group(self):
        with tempfile.TemporaryDirectory(prefix="ranking-fresh-", dir=CACHE) as temporary:
            directory = Path(temporary)
            documents = directory / "documents.jsonl"
            rows = [dict(row(group=f"{source}-{i}", source=source, split="holdout"), split_group=f"{source}-one-series")
                    for source in ("lccc", "naturalconv", "cped") for i in range(8)]
            write_rows(documents, rows)
            recipe = BUILD.build(documents, directory / "out", directory / "evidence", [])
            cases = read_rows(directory / "out/ranking-holdout.jsonl")
            self.assertEqual(set(recipe["splits"]["holdout"]["source"]), {"naturalconv", "cped"})
            self.assertGreater(len(cases), 2)
            self.assertTrue(all(case["source_group"].endswith("-one-series") for case in cases))
            self.assertEqual(len({case["document_group"] for case in cases}), len(cases))
            self.assertTrue(all(case["group_id"] == case["source_group"] for case in cases))
            self.assertEqual(recipe["splits"]["holdout"]["source_groups"]["cped"], 1)


if __name__ == "__main__":
    unittest.main()
