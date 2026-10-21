import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_general_corpus", ROOT / "scripts/prepare_general_corpus.py")
PREPARE = importlib.util.module_from_spec(SPEC)
if all(importlib.util.find_spec(name) for name in ("opencc", "pyarrow")):
    SPEC.loader.exec_module(PREPARE)
else:
    PREPARE = None
CACHE = ROOT / "data/cache/general-model-20261021"


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
                             [("9", "乳糖是一种糖。"), ("9", "人类的另一种答案。"), ("9", "同一问题的回答。")])

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


if __name__ == "__main__":
    unittest.main()
