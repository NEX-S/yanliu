import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_general_corpus", ROOT / "scripts/prepare_general_corpus.py")
PREPARE = importlib.util.module_from_spec(SPEC)
if all(importlib.util.find_spec(name) for name in ("opencc", "pyarrow")):
    SPEC.loader.exec_module(PREPARE)
else:
    PREPARE = None
CACHE = ROOT / "data/cache/retrain-20261021/test-fixtures"
CACHE.mkdir(parents=True, exist_ok=True)


@unittest.skipIf(PREPARE is None, "corpus preparation requires pyarrow and opencc-python-reimplemented")
class CorpusPreparationTests(unittest.TestCase):
    def test_chinese_normalization_preserves_daily_and_medical_language(self):
        value = "你 去 哪 里 了 ， 今天 有 空 吗 ？\n這是牛乳和母乳的區別。\n好\nPython 程序很好用。"
        self.assertEqual(list(PREPARE.clean_paragraphs(value)),
                         ["你去哪里了，今天有空吗？", "这是牛乳和母乳的区别。", "好", "Python 程序很好用。"])

    def test_filters_links_ads_and_code_without_whole_document_loss(self):
        value = "日常生活中喝水很重要。\n欢迎访问 https://example.com\n扫码关注领取奖励\nint value = call(a) { return 1; }\n参考文献\n乳房健康也很重要。"
        self.assertEqual(list(PREPARE.clean_paragraphs(value)), ["日常生活中喝水很重要。", "乳房健康也很重要。"])

    def test_long_paragraphs_reconstruct_original_text(self):
        value = "今天我们一起研究汉字的来源。" * 90
        pieces = list(PREPARE.clean_paragraphs(value))
        self.assertEqual("".join(pieces), value)
        self.assertTrue(all(len(piece) <= 384 for piece in pieces))

    def test_dialogue_keeps_turn_order_punctuation_and_filters_each_turn(self):
        dialogue = ["你 今天 去 哪里", "我去公園了！", "扫码关注领取奖励", "好", "你还要一起吃晚饭吗？"]
        self.assertEqual(PREPARE.clean_dialogue(dialogue), "你今天去哪里。我去公园了！好。你还要一起吃晚饭吗？")
        long = PREPARE.clean_dialogue(["今天我们一起去公园散步。"] * 100).split("\n\n")
        self.assertEqual("".join(long), "今天我们一起去公园散步。" * 100)
        self.assertTrue(all(len(piece) <= 384 for piece in long))

    def test_near_copy_anchors_catch_edits_and_do_not_collapse_unrelated_prose(self):
        original = "日常中文输入时经常需要根据前后文正确选择同音汉字。" * 5
        edited = original[:30] + "说明" + original[32:]
        unrelated = "周末我准备去公园散步和朋友聊天之后回家吃晚饭。" * 5
        self.assertTrue(set(PREPARE.near_keys(original)) & set(PREPARE.near_keys(edited)))
        self.assertTrue(PREPARE.copied_text(original, edited))
        self.assertFalse(set(PREPARE.near_keys(original)) & set(PREPARE.near_keys(unrelated)))
        self.assertEqual(PREPARE.near_keys("好的"), ())

    def test_prefixed_and_hard_wrapped_copy_has_stable_candidates_and_verified_overlap(self):
        original = PREPARE.CONVERT.convert("".join(chr(0x4E00 + i) for i in range(200)))
        prefixed = "注意" + original
        self.assertGreaterEqual(len(set(PREPARE.near_keys(original)) & set(PREPARE.near_keys(prefixed))), 2)
        self.assertTrue(PREPARE.copied_text(original, prefixed))
        self.assertEqual(list(PREPARE.clean_paragraphs(original[:90] + "\n" + original[90:])), [original])
        for cut in range(1, len(original)):
            largest = max(PREPARE.clean_paragraphs(original[:cut] + "\n" + original[cut:]), key=len)
            self.assertTrue(PREPARE.copied_text(PREPARE.canonical(original), PREPARE.canonical(largest)), cut)
            self.assertGreaterEqual(len(set(PREPARE.near_keys(PREPARE.canonical(original))) & set(PREPARE.near_keys(PREPARE.canonical(largest)))), 2, cut)
        unrelated = original[:24] + "".join(chr(0x6000 + i) for i in range(152)) + original[-24:]
        self.assertFalse(PREPARE.copied_text(original, unrelated))

    def test_whole_group_split_is_stable_before_paragraph_segmentation(self):
        self.assertEqual(PREPARE.group_split("wiki", "123"), PREPARE.group_split("wiki", "123"))
        assignments = Counter(PREPARE.group_split("wiki", str(i)) for i in range(10000))
        self.assertTrue(9400 < assignments["train"] < 9900)
        self.assertTrue(75 < assignments["dev"] < 230)
        self.assertTrue(75 < assignments["holdout"] < 230)

    def test_human_cqia_answers_keep_question_group_and_ignore_synthetic_answers(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-general-corpus-", dir=CACHE) as temporary:
            path = Path(temporary) / "cqia.jsonl"
            rows = [{"instruction": "什么是乳糖？", "output": "乳糖是一种糖。", "answer_from": "human", "metadata": {"qid": "9"}},
                    {"instruction": "什么是乳糖？", "output": "另一种答案。", "answer_from": "gpt", "metadata": {"qid": "9"}},
                    {"instruction": "乳糖是什么？", "output": "人类的另一种答案。", "answer_from": "human", "metadata": '{"question_id":"9"}'},
                    {"instruction": "你能解释乳糖吗？", "output": "同一问题的回答。", "answer_from": "human", "metadata": "qid:9, aid:123, tag:['知识']"}]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            self.assertEqual(list(PREPARE.source_records("cqia", [path])),
                             [("cqia:9", "cqia:9", PREPARE.group_split("cqia", "cqia:9"), text)
                              for text in ("乳糖是一种糖。", "人类的另一种答案。", "同一问题的回答。")])

    def test_cqia_namespaces_qids_but_groups_same_zhihu_question_across_files(self):
        with tempfile.TemporaryDirectory(prefix="cqia-groups-", dir=CACHE) as temporary:
            paths = [Path(temporary) / name for name in ("cqia-zhihu9.jsonl", "cqia-zhihu8.jsonl", "cqia-zgbk.jsonl")]
            for path in paths:
                path.write_text(json.dumps({"instruction": "问题", "output": "真实回答", "answer_from": "human", "metadata": {"qid": "9"}}, ensure_ascii=False) + "\n")
            records = list(PREPARE.source_records("cqia", paths))
            self.assertEqual([record[0] for record in records], ["zhihu:9", "zhihu:9", "zgbk:9"])
            self.assertEqual(records[0][1:3], records[1][1:3])

    def test_cqia_preparation_preserves_distinct_answers_in_one_question_group(self):
        with tempfile.TemporaryDirectory(prefix="cqia-prepare-", dir=CACHE) as temporary:
            directory = Path(temporary)
            sources = directory / "sources"
            sources.mkdir()
            path = sources / "cqia-zhihu9.jsonl"
            texts = ["乳糖是哺乳动物乳汁中的一种二糖。", "乳糖经过消化后可以分解成葡萄糖和半乳糖。"]
            path.write_text("".join(json.dumps({"instruction": "乳糖是什么？", "output": text, "answer_from": "human", "metadata": {"qid": "9"}}, ensure_ascii=False) + "\n" for text in texts + [texts[0]]))
            for number in (1, 2, 3):
                (sources / f"{number}.txt").write_text("乳糖")
            previous = directory / "previous.txt"
            previous.write_text("")
            def specs(output, *args):
                for name in ("compact", "small", "large"):
                    (output / f"{name}-spec.json").write_text("{}\n")
                return list(set("".join(texts))), {}
            args = ["prepare_general_corpus.py", "--root", str(directory), "--output", str(directory / "out"), "--evidence", str(directory / "evidence"), "--previous-train", str(previous)]
            files = {"cqia": [(path.name, "public/repo", "fixed-revision", "file.jsonl", None)]}
            with mock.patch.object(PREPARE, "SOURCE_FILES", files), mock.patch.object(PREPARE, "make_specs", side_effect=specs), mock.patch("sys.argv", args), mock.patch("builtins.print"):
                PREPARE.main()
            documents = [json.loads(line) for line in (directory / "out/documents.jsonl").read_text().splitlines()]
            self.assertEqual({row["text"] for row in documents}, set(texts))
            self.assertEqual({row["group"] for row in documents}, {"zhihu:9"})
            self.assertEqual([row["paragraph"] for row in documents], [0, 1])
            self.assertEqual(len({row["split"] for row in documents}), 1)
            exclusion = directory / "observed-holdout.jsonl"
            exclusion.write_text(json.dumps({"input": "rutang", "context": texts[0][:6], "gold": texts[0][6:]}, ensure_ascii=False) + "\n")
            excluded_args = ["prepare_general_corpus.py", "--root", str(directory), "--output", str(directory / "excluded"), "--evidence", str(directory / "excluded-evidence"), "--previous-train", str(previous), "--exclude", str(exclusion)]
            with mock.patch.object(PREPARE, "SOURCE_FILES", files), mock.patch.object(PREPARE, "make_specs", side_effect=specs), mock.patch("sys.argv", excluded_args), mock.patch("builtins.print"):
                PREPARE.main()
            excluded_documents = [json.loads(line) for line in (directory / "excluded/documents.jsonl").read_text().splitlines()]
            self.assertEqual([row["text"] for row in excluded_documents], texts[1:])
            recipe = json.loads((directory / "excluded/data-recipe.json").read_text())
            self.assertIn({"path": str(exclusion), "sha256": PREPARE.digest(exclusion)}, recipe["eval_exclusion_files"])
            missing = directory / "missing.jsonl"
            missing_args = ["prepare_general_corpus.py", "--root", str(directory), "--output", str(directory / "missing-out"), "--evidence", str(directory / "missing-evidence"), "--previous-train", str(previous), "--exclude", str(missing)]
            with mock.patch.object(PREPARE, "SOURCE_FILES", files), mock.patch("sys.argv", missing_args):
                with self.assertRaises(FileNotFoundError) as error:
                    PREPARE.main()
            self.assertEqual(error.exception.filename, str(missing))
            self.assertFalse((directory / "missing-out").exists())
            self.assertFalse((directory / "missing-evidence").exists())

    def test_new_sources_preserve_conversations_and_inherit_larger_split_groups(self):
        with tempfile.TemporaryDirectory(prefix="new-sources-", dir=CACHE) as temporary:
            directory = Path(temporary)
            path = directory / "naturalconv.json"
            path.write_text(json.dumps([{"document_id": 0, "dialog_id": "0_0", "content": ["先前看过的新闻"]},
                                        {"document_id": 8, "dialog_id": "8_0", "content": ["你好", "我也很好！"]},
                                        {"document_id": 8, "dialog_id": "8_1", "content": ["我们去公园吧", "好呀"]}], ensure_ascii=False))
            records = list(PREPARE.source_records("naturalconv", [path]))
            self.assertEqual([r[0] for r in records], ["8_0", "8_1"])
            self.assertEqual(records[0][1:3], records[1][1:3])
            self.assertEqual([r[3] for r in records], ["你好。我也很好！", "我们去公园吧。好呀。"])
            path = directory / "cped-valid.csv"
            path.write_text("TV_ID,Dialogue_ID,Utterance_ID,Utterance\n32,32_000,32_000_001,第二句话\n32,32_000,32_000_000,第一句话\n32,32_001,32_001_000,另一次对话\n", encoding="utf-8-sig")
            records = list(PREPARE.source_records("cped", [path]))
            self.assertEqual(records, [("32_000", "32", "dev", "第一句话。第二句话。"), ("32_001", "32", "dev", "另一次对话。")])
            other = directory / "cped-test.csv"
            other.write_text(path.read_text(encoding="utf-8-sig"))
            with self.assertRaisesRegex(ValueError, "crosses official splits"):
                list(PREPARE.source_records("cped", [path, other]))

    def test_previous_training_index_uses_same_exact_and_near_rules(self):
        with tempfile.TemporaryDirectory(prefix="previous-train-", dir=CACHE) as temporary:
            connection = PREPARE.open_dedup(Path(temporary) / "seen.sqlite3")
            original = PREPARE.canonical("日常中文输入时经常需要根据前后文正确选择同音汉字。" * 5)
            PREPARE.remember_text(connection, original, PREPARE.near_keys(original))
            self.assertEqual(PREPARE.duplicate_reason(connection, original, PREPARE.near_keys(original)), "exact")
            edited = original[:30] + "说明" + original[32:]
            self.assertEqual(PREPARE.duplicate_reason(connection, edited, PREPARE.near_keys(edited)), "near")
            self.assertIsNone(PREPARE.duplicate_reason(connection, "今天去公园吃饭聊天", ()))
            connection.close()

    def test_previous_training_filters_fresh_evaluation_but_allows_training_reuse(self):
        with tempfile.TemporaryDirectory(prefix="fresh-overlap-", dir=CACHE) as temporary:
            directory = Path(temporary)
            sources = directory / "sources"
            sources.mkdir()
            shared = "今天我们一起去公园讨论工作之后回家吃饭。"
            copied = "日常中文输入时经常需要根据前后文正确选择同音汉字。" * 5
            near = copied[:30] + "说明" + copied[32:]
            fresh = "周末我准备去博物馆看看新开办的历史展览。"
            rows = {"train": [("1", "01_000", shared)], "valid": [("32", "32_000", copied)],
                    "test": [("27", "27_000", shared), ("27", "27_001", near), ("27", "27_002", fresh)]}
            for split, records in rows.items():
                path = sources / f"cped-{split}.csv"
                path.write_text("TV_ID,Dialogue_ID,Utterance_ID,Utterance\n" + "".join(f"{tv},{dialogue},{dialogue}_000,{text}\n" for tv, dialogue, text in records), encoding="utf-8-sig")
            for number in (1, 2, 3):
                (sources / f"{number}.txt").write_text("字")
            previous = directory / "previous.txt"
            previous.write_text(shared + "\n" + copied + "\n")
            def specs(output, *args):
                for name in ("compact", "small", "large"):
                    (output / f"{name}-spec.json").write_text("{}\n")
                return list(set(shared + fresh)), {}
            args = ["prepare_general_corpus.py", "--root", str(directory), "--output", str(directory / "out"), "--evidence", str(directory / "evidence"), "--previous-train", str(previous)]
            files = {"cped": [(f"cped-{split}.csv", "public/repo", "fixed", "file.csv", None) for split in rows]}
            with mock.patch.object(PREPARE, "SOURCE_FILES", files), mock.patch.object(PREPARE, "make_specs", side_effect=specs), mock.patch("sys.argv", args), mock.patch("builtins.print"):
                PREPARE.main()
            self.assertEqual((directory / "out/train.txt").read_text(), shared + "\n")
            self.assertEqual((directory / "out/dev.txt").read_text(), "")
            self.assertEqual((directory / "out/holdout.txt").read_text(), fresh + "\n")
            recipe = json.loads((directory / "out/data-recipe.json").read_text())
            stats = recipe["source_stats"]["cped"]
            self.assertEqual(stats["dev_previous_train_exact_removed"], 1)
            self.assertEqual(stats["holdout_previous_train_exact_removed"], 1)
            self.assertEqual(stats["holdout_previous_train_near_removed"], 1)
            self.assertEqual(recipe["previous_train"]["matching_fresh_paragraphs"], 3)

    def test_required_characters_are_independent_of_training_frequency(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-general-vocabulary-", dir=CACHE) as temporary:
            sources = Path(temporary)
            chars = [chr(0x4E00 + i) for i in range(8105)]
            for number, table in enumerate((chars[:3500], chars[3500:6500], chars[6500:]), 1):
                (sources / f"{number}.txt").write_text("\n".join(table))
            dictionary = sources / "dictionary.yaml"
            dictionary.write_text("乳\tru\t100\n𬭚\tdui\t1\n〇\tling\t1\n")
            vocab, coverage = PREPARE.vocabulary(sources, dictionary, Counter({"🙂": 12, "🫠": 1}))
            with self.assertRaisesRegex(ValueError, "coverage exceeds"):
                PREPARE.vocabulary(sources, dictionary, Counter(), maximum=100)
        self.assertEqual(coverage["required8105"], 8105)
        self.assertEqual(coverage["dictionary_single_characters"], coverage["dictionary_single_characters_covered"])
        self.assertIn("乳", vocab)
        self.assertIn("𬭚", vocab)
        self.assertIn("〇", vocab)
        self.assertIn("🙂", vocab)
        self.assertNotIn("🫠", vocab)
        self.assertEqual(vocab[:3], ["<pad>", "<unk>", "<bos>"])
        self.assertEqual(len(vocab), len(set(vocab)))

    def test_public_evaluation_exclusion_accepts_single_and_multiple_golds(self):
        with tempfile.TemporaryDirectory(prefix="test-general-exclusion-", dir=CACHE) as temporary:
            path = Path(temporary) / "eval.jsonl"
            rows = [{"context": "他说今天", "gold": "天气很好"},
                    {"context": "", "golds": ["我生病了不是吗", "我今天没有生病"]},
                    {"context": "", "gold": "对"}]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            values, files = PREPARE.evaluation_exclusions([path])
            self.assertEqual(values, {"他说今天天气很好", "我生病了不是吗", "我今天没有生病"})
            self.assertEqual(files[0]["sha256"], PREPARE.digest(path))

    def test_public_learning_choices_obey_the_same_text_exclusion_length_rule(self):
        values, _ = PREPARE.evaluation_exclusions([ROOT / "data/eval/learning-context.jsonl"])
        self.assertEqual(values, {"乘火车前往西安"})


if __name__ == "__main__":
    unittest.main()
