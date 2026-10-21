#!/usr/bin/env python3
"""Prepare the fixed 2026-10-21 public Chinese corpus; never reads local conversations."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

import opencc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONVERT = opencc.OpenCC("t2s")
HAN = re.compile(r"[\u3007\u3400-\u9fff\U00020000-\U000323af]")
CANONICAL = re.compile(r"[^\w\u3400-\u9fff\U00020000-\U000323af]", re.UNICODE)
URL = re.compile(r"https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:wx|微信|QQ)[:：号 ]*[a-zA-Z0-9_-]{5,}", re.I)
NAV = re.compile(r"^(?:参考文献|外部链接|参见|相关条目|延伸阅读|参考资料|注释|脚注|参阅|参考来源|资料来源|引用|注解)$")
AD = re.compile(r"(?:加[我微].{0,3}(?:信|号)|扫码关注|点击领取|转发抽奖|购买链接|备用网址|博彩|代开发票|免费领红包)")
MARKDOWN_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]*\)")
PUNCTUATION = "，。！？；：、‘’“”（）【】《》〈〉〔〕［］｛｝—…·～￥「」『』﹏﹑﹔﹕﹖﹗﹙﹚﹛﹜﹝﹞"
SOURCE_FILES = {
    "naturalconv": [("naturalconv-dialog.json", "xywang1/NaturalConv", "adb38574e8014ada1d8404247167582e76b22d05", "dialog_release.json", "0aed101b0f295f87da8d27c3fb4eb62955488b7116c9a9c63c929f65656c195a")],
    "cped": [(f"cped-{split}.csv", "scutcyr/CPED", "1e4b81c28a123f22387e06664f37e5dc9322380f", f"data/CPED/{split}_split.csv", expected)
             for split, expected in (("train", "7bce6a11f0bf0ac353d28e96f6cfa03ff5e83b95810647abb55d18ff7299e487"),
                                     ("valid", "f434a04d21640bb38c849f3e9598c2b7b2fbdb49be6a7365b2ae016019bd389b"),
                                     ("test", "a4bcde678438ba1f95224d8e772e40f1f1df7c72f348796b3fafd421c2443a53"))],
    "lccc": [("lccc-train-00000.parquet", "thu-coai/lccc", "81d998c10513b24cd560be3448d693e80d753fdc", "base/lccc-train-00000-of-00002.parquet", "1e8bbfff077ec1a7c74672d156f93b1d443ed8f240383ac3a317eb2afa6a5e5f")],
    "wiki": [("wiki-train-00000.parquet", "wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.zh/train-00000-of-00006.parquet", "853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56"),
             ("wiki-train-00002.parquet", "wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.zh/train-00002-of-00006.parquet", "ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f")],
    "cqia": [("cqia-zhihu9.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "zhihu/zhihu_score9.0-10_clean_v10.jsonl", None),
             ("cqia-zhihu85.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "zhihu/zhihu_score8.5-9.0_clean_v4.jsonl", None),
             ("cqia-zhihu8.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "zhihu/zhihu_score8.0-8.5_clean_v3.jsonl", None),
             ("cqia-wikihow.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "wikihow/wikihow.jsonl", "ec172b8616c589bc28b5be5c77ae4b0659e401184848127517bc8352497d57ca"),
             ("cqia-zgbk.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "wiki/zgbk.jsonl", None),
             ("cqia-book-introduce.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "douban/book_introduce.jsonl", None),
             ("cqia-book-reviews.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "douban/book_reviews.jsonl", None),
             ("cqia-movie-reviews.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "douban/movie_reviews.jsonl", None),
             ("cqia-tv-reviews.jsonl", "m-a-p/COIG-CQIA", "8b55868c6168adf86c30e7ca0f782cca1c514297", "douban/tv_reviews.jsonl", None)],
    "tnews": [("tnews-train.parquet", "clue/clue", "28178267a609dd08bdc703dd6c931dfc2c2f4431", "tnews/train-00000-of-00001.parquet", "57ad8b1a5a6a4120a283468f5a828a2702b742139a1878e1edcd7b8b59eec5de")],
}
ATTRIBUTIONS = {
    "naturalconv": {"declared_license": "not declared in dataset card; retain original source", "data_card": "https://huggingface.co/datasets/xywang1/NaturalConv/tree/adb38574e8014ada1d8404247167582e76b22d05", "citation": "Wang et al. (2021), NaturalConv: A Chinese Dialogue Dataset Towards Multi-turn Topic-driven Conversation; https://arxiv.org/abs/2103.02548"},
    "cped": {"declared_license": "Apache-2.0 repository license; original TV dialogue source retained", "license_url": "https://raw.githubusercontent.com/scutcyr/CPED/1e4b81c28a123f22387e06664f37e5dc9322380f/LICENSE", "data_card": "https://github.com/scutcyr/CPED/tree/1e4b81c28a123f22387e06664f37e5dc9322380f", "citation": "Chen et al. (2022), CPED: A Large-Scale Chinese Personalized and Emotional Dialogue Dataset for Conversational AI; text only, no speaker or visual annotations"},
    "lccc": {"declared_license": "MIT", "data_card": "https://huggingface.co/datasets/thu-coai/lccc/raw/9022ba27075f75c2f59d57d7fa5f42e8d1151aec/README.md", "citation": "Wang, Yida et al. (2020), A Large-Scale Chinese Short-Text Conversation Dataset, NLPCC; https://arxiv.org/abs/2008.03946"},
    "wiki": {"declared_license": "CC-BY-SA-3.0 and GFDL (version not specified in card)", "data_card": "https://huggingface.co/datasets/wikimedia/wikipedia/raw/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md", "citation": "Wikimedia Wikipedia 2023-11-01 Chinese dump; original article id/url/title retained in raw Parquet"},
    "cqia": {"declared_license": "More Information Needed; no unified license declared; per-row copyright retained in raw JSONL", "data_card": "https://huggingface.co/datasets/m-a-p/COIG-CQIA/raw/8b55868c6168adf86c30e7ca0f782cca1c514297/README.md", "citation": "Bai, Yuelin et al. (2024), COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning; https://arxiv.org/abs/2403.18058"},
    "tnews": {"declared_license": "unknown", "data_card": "https://huggingface.co/datasets/clue/clue/raw/28178267a609dd08bdc703dd6c931dfc2c2f4431/README.md", "citation": "Xu, Liang et al. (2020), CLUE: A Chinese Language Understanding Evaluation Benchmark; https://arxiv.org/abs/2004.05986"},
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical(text):
    return CANONICAL.sub("", unicodedata.normalize("NFKC", text)).lower()


def fingerprint(text):
    return hashlib.blake2b(text.encode(), digest_size=12).digest()


def group_split(source, group):
    number = int.from_bytes(fingerprint(source + "\0" + group)[:4], "big") % 10000
    if source == "naturalconv":
        return "dev" if number < 1000 else "holdout" if number < 2000 else "train"
    return "dev" if number < 150 else "holdout" if number < 300 else "train"


def clean_paragraphs(text):
    """Keep ordinary short Chinese, including medical vocabulary; reject code and ads."""
    text = CONVERT.convert(unicodedata.normalize("NFC", text))
    text = MARKDOWN_LINK.sub(r"\1", text)
    text = re.sub(r"<[^>]{0,300}>", "", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if (lines and len(lines[-1]) >= 12 and len(line) >= 12
                and lines[-1][-1] not in "。！？.!?;；：:" and not line.startswith(("#", "*", "-", "+"))):
            lines[-1] += line
        else:
            lines.append(line)
    for line in lines:
        if URL.search(line) or AD.search(line):
            continue
        line = re.sub(r"^\s*(?:#{1,6}\s*|(?:[-*+] |\d+[.、] ))", "", line)
        line = line.replace("**", "").replace("__", "").strip(" \t\r\n")
        line = "".join(char for char in line if unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Co"})
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fff])", "", line)
        line = re.sub(r"\s*([，。！？；：、‘’“”（）【】《》,.!?;:])\s*", r"\1", line)
        if not line or NAV.fullmatch(line) or "```" in line:
            continue
        han = len(HAN.findall(line))
        if han < 1 or han / max(1, sum(char.isalnum() for char in line)) < 0.45:
            continue
        if re.search(r"(.)\1{5,}", line) or len(re.findall(r"[{}=<>\\|]", line)) > max(2, len(line) // 30):
            continue
        yield from split_paragraph(line)


def split_paragraph(line):
    while len(line) > 384:
        cut = max(line.rfind(char, 160, 384) for char in "。！？；.!?;") + 1
        if cut < 160:
            cut = 384
        yield line[:cut]
        line = line[cut:].lstrip()
    if line:
        yield line


def clean_dialogue(utterances):
    turns = []
    for utterance in utterances:
        text = "".join(clean_paragraphs(utterance))
        if not text:
            continue
        if text.rstrip("\"'”’」』）)")[-1:] not in "。！？.!?；;…，,：:":
            text += "。"
        turns.append(text)
    return "\n\n".join(split_paragraph("".join(turns)))


def near_keys(text):
    """Content-selected shingles survive a small prefix or suffix edit."""
    if len(text) < 64:
        return ()
    return tuple(sorted({hashlib.blake2b(text[start:start + 16].encode(), digest_size=8).digest()
                         for start in range(len(text) - 15)})[:12])


def copied_text(first, second):
    if min(len(first), len(second)) < 0.9 * max(len(first), len(second)):
        return False
    left = Counter(first[start:start + 4] for start in range(len(first) - 3))
    right = Counter(second[start:start + 4] for start in range(len(second) - 3))
    return sum((left & right).values()) >= 0.9 * max(len(first) - 3, len(second) - 3)


def open_dedup(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE fingerprints(key BLOB PRIMARY KEY) WITHOUT ROWID")
    connection.execute("CREATE TABLE texts(id INTEGER PRIMARY KEY, text TEXT)")
    connection.execute("CREATE TABLE shingles(key BLOB, document INTEGER, PRIMARY KEY(key,document)) WITHOUT ROWID")
    return connection


def duplicate_reason(connection, normalized, keys):
    if connection.execute("SELECT 1 FROM fingerprints WHERE key=?", (fingerprint(normalized),)).fetchone():
        return "exact"
    if keys:
        placeholders = ",".join("?" for _ in keys)
        matches = connection.execute(f"SELECT document FROM shingles WHERE key IN ({placeholders}) GROUP BY document HAVING COUNT(*)>=2", keys)
        for (document,) in matches:
            previous = connection.execute("SELECT text FROM texts WHERE id=?", (document,)).fetchone()[0]
            if copied_text(normalized, previous):
                return "near"
    return None


def remember_text(connection, normalized, keys):
    connection.execute("INSERT OR IGNORE INTO fingerprints VALUES(?)", (fingerprint(normalized),))
    if keys:
        document = connection.execute("INSERT INTO texts(text) VALUES(?)", (normalized,)).lastrowid
        connection.executemany("INSERT INTO shingles VALUES(?,?)", ((value, document) for value in keys))


def source_records(source, paths):
    cped_ownership = {}
    for path in paths:
        if source == "naturalconv":
            for row in json.loads(path.read_text(encoding="utf-8")):
                group = str(row["document_id"])
                if row["document_id"] in range(7):
                    continue
                yield row["dialog_id"], group, group_split(source, group), clean_dialogue(row["content"])
        elif source == "cped":
            split = {"cped-train": "train", "cped-valid": "dev", "cped-test": "holdout"}[path.stem]
            conversations = defaultdict(list)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    tv = row["TV_ID"]
                    if tv in cped_ownership and cped_ownership[tv] != split:
                        raise ValueError(f"{path}: TV_ID {tv} crosses official splits")
                    cped_ownership[tv] = split
                    conversations[(tv, row["Dialogue_ID"])].append((row["Utterance_ID"], row["Utterance"]))
            for (tv, dialogue), turns in conversations.items():
                yield dialogue, tv, split, clean_dialogue(text for _, text in sorted(turns))
        elif path.suffix == ".parquet":
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    if source == "lccc":
                        group = fingerprint(canonical("\n\n".join(row["dialog"]))).hex()
                        text = clean_dialogue(row["dialog"])
                    elif source == "wiki":
                        text = re.split(r"\n\s*(?:参考文献|外部链接|延伸阅读|参考资料|注释|脚注|参考来源|资料来源)[:：]?\s*\n", row["text"], maxsplit=1)[0]
                        group = str(row["id"])
                    else:
                        group, text = str(row["idx"]), row["sentence"]
                    yield group, group, group_split(source, group), text
        else:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("answer_from") != "human":
                        continue
                    metadata = row.get("metadata")
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            qid = re.search(r"(?:^|[,;]\s*)qid\s*[:：]\s*(\d+)", metadata)
                            metadata = {"qid": qid[1]} if qid else {}
                    metadata = metadata if isinstance(metadata, dict) else {}
                    question = canonical(CONVERT.convert(unicodedata.normalize("NFC", row["instruction"])))
                    namespace = "zhihu" if path.stem.startswith("cqia-zhihu") else path.stem.removeprefix("cqia-")
                    group = namespace + ":" + str(metadata.get("qid") or metadata.get("question_id") or fingerprint(question).hex())
                    # CQIA sometimes synthesizes instructions around human encyclopedia prose.
                    yield group, group, group_split(source, group), row["output"]


def evaluation_exclusions(paths):
    values = set()
    files = []
    for path in paths:
        files.append({"path": str(path), "sha256": digest(path)})
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                selected = row.get("select")
                target = selected.get("text") if isinstance(selected, dict) else None
                golds, context = row.get("golds", [row.get("gold", target)]), row.get("context", "")
                if isinstance(golds, list) and isinstance(context, str):
                    for gold in golds:
                        if isinstance(gold, str):
                            text = canonical(CONVERT.convert(unicodedata.normalize("NFC", context + gold)))
                            if len(text) >= 6:
                                values.add(text)
    return values, files


def vocabulary(sources, dictionary, frequencies, maximum=10240):
    tables = [[char for char in (sources / f"{i}.txt").read_text(encoding="utf-8") if HAN.fullmatch(char)] for i in (1, 2, 3)]
    if [len(set(table)) for table in tables] != [3500, 3000, 1605] or len(set(sum(tables, []))) != 8105:
        raise ValueError("the three character tables must contain 3500+3000+1605 disjoint characters")
    dictionary_chars = set()
    with dictionary.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            char = line.split("\t", 1)[0]
            if len(char) == 1 and HAN.fullmatch(char):
                dictionary_chars.add(char)
    ordered = sum(tables, []) + sorted(dictionary_chars) + [chr(code) for code in range(32, 127)] + list(PUNCTUATION)
    vocab = list(dict.fromkeys(["<pad>", "<unk>", "<bos>"] + ordered))
    if len(vocab) > maximum:
        raise ValueError("required character coverage exceeds the vocabulary limit")
    for char, count in frequencies.most_common():
        if count < 8 or len(vocab) == maximum:
            break
        if char not in vocab and unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Co"}:
            vocab.append(char)
    return vocab, {"required3500": len(set(tables[0]) & set(vocab)),
                   "required6500": len(set(tables[0] + tables[1]) & set(vocab)),
                   "required8105": len(set(sum(tables, [])) & set(vocab)),
                   "dictionary_single_characters": len(dictionary_chars),
                   "dictionary_single_characters_covered": len(dictionary_chars & set(vocab)),
                   **{f"train_observed{size}": sum(frequencies[char] > 0 for char in chars)
                      for size, chars in ((3500, tables[0]), (6500, tables[0] + tables[1]), (8105, sum(tables, [])))},
                   **{f"train_atleast100_{size}": sum(frequencies[char] >= 100 for char in chars)
                      for size, chars in ((3500, tables[0]), (6500, tables[0] + tables[1]), (8105, sum(tables, [])))}}


def make_specs(output, sources, dictionary, frequencies, attribution):
    vocab, coverage = vocabulary(sources, dictionary, frequencies)
    provenance = {"license": "Mixed public sources; per-source attribution retained in data-recipe.json",
                  "data_classification": "public Chinese text, grouped held-out split; no private conversations or system learning",
                  "attribution": attribution}
    for name, layers, width, heads in (("compact", 6, 192, 4), ("small", 6, 256, 4), ("large", 8, 448, 8)):
        write_json(output / f"{name}-spec.json", {"config": {"vocab": len(vocab), "n_layer": layers,
                   "n_head": heads, "n_embd": width, "context": 64, "dropout": 0.1},
                   "vocab": vocab, "provenance": provenance})
    return vocab, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/cache/retrain-20261021")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--lccc-chars", type=int, default=15_000_000)
    parser.add_argument("--wiki-chars", type=int, default=25_000_000)
    parser.add_argument("--cqia-chars", type=int, default=12_000_000)
    parser.add_argument("--tnews-chars", type=int, default=1_000_000)
    parser.add_argument("--naturalconv-chars", type=int, default=2_000_000)
    parser.add_argument("--cped-chars", type=int, default=1_000_000)
    parser.add_argument("--previous-train", type=Path, required=True,
                        help="previous model training text, excluded only from fresh dev/holdout by exact/near paragraph matching")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="additional observed evaluation JSONL to exclude from every corpus split; may be repeated")
    args = parser.parse_args()
    converter = {"package": "OpenCC", "version": importlib.metadata.version("OpenCC"),
                 "module": opencc.__file__, "configuration": "t2s"}
    limits = {key: getattr(args, key + "_chars") for key in SOURCE_FILES}
    if min(limits.values()) < 1:
        raise ValueError("character budgets must be positive")
    sources = args.root / "sources"
    manifest = []
    for source, files in SOURCE_FILES.items():
        for filename, repo, revision, remote, expected in files:
            path = sources / filename
            actual = digest(path)
            if expected is not None and actual != expected:
                raise ValueError(f"{path}: SHA256 mismatch: expected {expected}, got {actual}")
            manifest.append({"source": source, "path": str(path), "sha256": actual, "bytes": path.stat().st_size,
                             "url": (f"https://raw.githubusercontent.com/{repo}/{revision}/{remote}" if source == "cped"
                                     else f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}"), "revision": revision,
                             **ATTRIBUTIONS[source]})
    exclusions, exclusion_files = evaluation_exclusions(sorted((ROOT / "data/eval").glob("*.jsonl")) + args.exclude)
    exclusion_pattern = re.compile("|".join(re.escape(value) for value in sorted(exclusions))) if exclusions else None
    args.output.mkdir(parents=True, exist_ok=False)
    args.evidence.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(Counter)
    stage_path = args.output / "staging.sqlite3"
    stage = sqlite3.connect(stage_path)
    stage.execute("PRAGMA journal_mode=OFF")
    stage.execute("PRAGMA synchronous=OFF")
    stage.execute("CREATE TABLE docs(source TEXT, group_id TEXT, split_group TEXT, split TEXT, shuffle BLOB, record BLOB, paragraphs TEXT, PRIMARY KEY(source,group_id,record)) WITHOUT ROWID")
    frozen_groups = defaultdict(dict)
    for source, files in SOURCE_FILES.items():
        for group, split_group, split, text in source_records(source, [sources / item[0] for item in files]):
            stats[source]["raw_records"] += 1
            if source in {"naturalconv", "cped"}:
                previous = frozen_groups[source].setdefault(split_group, split)
                if previous != split:
                    raise ValueError(f"{source}:{split_group}: source group crosses splits")
            paragraphs = text.split("\n\n") if source in {"lccc", "naturalconv", "cped"} and text else list(clean_paragraphs(text))
            if not paragraphs:
                stats[source]["empty_after_cleaning"] += 1
                continue
            before = stage.total_changes
            stage.execute("INSERT OR IGNORE INTO docs VALUES(?,?,?,?,?,?,?)", (source, group, split_group, split, fingerprint(source + "\0" + group), fingerprint(canonical(text)), json.dumps(paragraphs, ensure_ascii=False)))
            if before == stage.total_changes:
                stats[source]["identical_record_in_group"] += 1
            if stats[source]["raw_records"] % 50000 == 0:
                stage.commit()
                print(json.dumps({"stage": "clean", "source": source, "raw_records": stats[source]["raw_records"]}), flush=True)
        stage.commit()
        print(json.dumps({"stage": "clean_done", "source": source, "stats": stats[source]}), flush=True)
    stage.execute("CREATE INDEX split_order ON docs(split,shuffle)")
    write_json(args.output / "source-groups.json", frozen_groups)
    previous_manifest = {"path": str(args.previous_train.resolve()), "sha256": digest(args.previous_train)}
    reference = open_dedup(args.output / "fresh-overlap.sqlite3")
    for (raw,) in stage.execute("SELECT paragraphs FROM docs WHERE source IN ('naturalconv','cped') AND split!='train'"):
        for text in json.loads(raw):
            normalized = canonical(text)
            if len(normalized) >= 12:
                remember_text(reference, normalized, near_keys(normalized))
    reference.commit()
    reference_fingerprints = {row[0] for row in reference.execute("SELECT key FROM fingerprints")}
    reference_shingles = {row[0] for row in reference.execute("SELECT DISTINCT key FROM shingles")}
    previous_overlaps = {}
    previous_manifest["indexed_fresh_paragraphs"] = len(reference_fingerprints)
    with args.previous_train.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            normalized = canonical(line.strip())
            exact = fingerprint(normalized)
            if exact in reference_fingerprints:
                previous_overlaps[exact] = "exact"
            keys = tuple(key for key in near_keys(normalized) if key in reference_shingles)
            if len(keys) >= 2:
                placeholders = ",".join("?" for _ in keys)
                matches = reference.execute(f"SELECT document FROM shingles WHERE key IN ({placeholders}) GROUP BY document HAVING COUNT(*)>=2", keys)
                for (document,) in matches:
                    previous = reference.execute("SELECT text FROM texts WHERE id=?", (document,)).fetchone()[0]
                    if copied_text(normalized, previous):
                        previous_overlaps.setdefault(fingerprint(previous), "near")
            if number % 100000 == 0:
                print(json.dumps({"stage": "previous_train_scan", "lines": number}), flush=True)
    previous_manifest["matching_fresh_paragraphs"] = len(previous_overlaps)
    reference.close()
    (args.output / "fresh-overlap.sqlite3").unlink()
    del reference_fingerprints, reference_shingles
    if digest(args.previous_train) != previous_manifest["sha256"]:
        raise ValueError("previous training text changed while indexing")
    seen = open_dedup(args.output / "dedup.sqlite3")
    short_counts = Counter()
    frequencies = Counter()
    source_frequencies = defaultdict(Counter)
    source_train_frequencies = defaultdict(Counter)
    source_lengths = defaultdict(Counter)
    split_totals = defaultdict(Counter)
    source_budgets = {(source, split): limit if split == "train" or source in {"naturalconv", "cped"} else int(limit * 0.015)
                      for source, limit in limits.items() for split in ("train", "dev", "holdout")}
    docs_path = args.output / "documents.jsonl"
    with docs_path.open("x", encoding="utf-8") as docs, \
            (args.output / "train.txt").open("x", encoding="utf-8") as train, \
            (args.output / "dev.txt").open("x", encoding="utf-8") as dev, \
            (args.output / "holdout.txt").open("x", encoding="utf-8") as holdout, \
            (args.output / "paragraph-map.jsonl").open("x", encoding="utf-8") as mapping:
        outputs = {"train": train, "dev": dev, "holdout": holdout}
        # Training first prevents a held-out copy of a training paragraph from surviving.
        for split in ("train", "dev", "holdout"):
            previous_group, paragraph_number = None, 0
            for source, group, split_group, raw in stage.execute("SELECT source,group_id,split_group,paragraphs FROM docs WHERE split=? ORDER BY shuffle,source,group_id,record", (split,)):
                if previous_group != (source, group):
                    previous_group, paragraph_number = (source, group), 0
                key = (source, split)
                if stats[source][split + "_characters"] >= source_budgets[key]:
                    continue
                accepted = []
                accepted_characters = 0
                for text in json.loads(raw):
                    if stats[source][split + "_characters"] + accepted_characters + len(text) > source_budgets[key]:
                        stats[source][split + "_budget_paragraph_not_selected"] += 1
                        continue
                    normalized = canonical(text)
                    if exclusion_pattern is not None and exclusion_pattern.search(normalized):
                        stats[source]["eval_overlap_removed"] += 1
                        continue
                    if len(normalized) < 12:
                        if split != "train" or short_counts[normalized] >= 32:
                            stats[source]["short_repeat_or_eval_removed"] += 1
                            continue
                        short_counts[normalized] += 1
                    else:
                        keys = near_keys(normalized)
                        if split != "train" and source in {"naturalconv", "cped"}:
                            reason = previous_overlaps.get(fingerprint(normalized))
                            if reason:
                                stats[source][split + "_previous_train_" + reason + "_removed"] += 1
                                continue
                        reason = duplicate_reason(seen, normalized, keys)
                        if reason:
                            stats[source]["duplicate_" + reason + "_removed"] += 1
                            continue
                        remember_text(seen, normalized, keys)
                    accepted.append(text)
                    accepted_characters += len(text)
                if not accepted:
                    continue
                if paragraph_number == 0:
                    stats[source][split + "_groups"] += 1
                for text in accepted:
                    row = {"source": source, "group": group, "split_group": split_group, "paragraph": paragraph_number, "split": split, "text": text}
                    docs.write(json.dumps(row, ensure_ascii=False) + "\n")
                    outputs[split].write(text + "\n")
                    split_totals[split]["paragraphs"] += 1
                    split_totals[split]["characters"] += len(text)
                    mapping.write(json.dumps({"split": split, "line": split_totals[split]["paragraphs"], "source": source, "group": group, "split_group": split_group, "paragraph": paragraph_number}, ensure_ascii=False) + "\n")
                    paragraph_number += 1
                    stats[source][split + "_characters"] += len(text)
                    stats[source][split + "_paragraphs"] += 1
                    source_frequencies[source].update(text)
                    source_lengths[source][len(text)] += 1
                    if split == "train":
                        frequencies.update(text)
                        source_train_frequencies[source].update(text)
                if split_totals[split]["paragraphs"] % 50000 < len(accepted):
                    seen.commit()
                    print(json.dumps({"stage": "select", "split": split, "totals": split_totals[split]}), flush=True)
            seen.commit()
    vocab, coverage = make_specs(args.output, sources, ROOT / "data/dictionaries/8105.dict.yaml", frequencies,
                                "LCCC-base, Wikimedia Chinese Wikipedia 20231101, COIG-CQIA human answers, CLUE TNEWS, NaturalConv and CPED dialogue text; fixed revisions and hashes in data-recipe.json")
    vocab_set = set(vocab)
    oov = {source: {"characters": sum(counts.values()), "unknown": sum(n for char, n in counts.items() if char not in vocab_set)} for source, counts in source_frequencies.items()}
    lengths = {}
    for source, counts in source_lengths.items():
        ordered = sorted(counts.items())
        total = sum(counts.values())
        quantiles = {}
        for percentile in (50, 95, 99, 100):
            cumulative = 0
            for length, count in ordered:
                cumulative += count
                if cumulative * 100 >= total * percentile:
                    quantiles[f"p{percentile}"] = length
                    break
        lengths[source] = {"paragraphs": total, "mean": sum(length * n for length, n in counts.items()) / total, **quantiles}
    required = [set(char for char in (sources / f"{i}.txt").read_text(encoding="utf-8") if HAN.fullmatch(char)) for i in (1, 2, 3)]
    source_training_coverage = {source: {"characters": sum(counts.values()), "distinct_han": sum(bool(HAN.fullmatch(char)) for char in counts),
                              **{f"observed{size}": len(set(counts) & chars) for size, chars in
                                 ((3500, required[0]), (6500, required[0] | required[1]), (8105, set.union(*required)))}}
                              for source, counts in source_train_frequencies.items()}
    recipe = {"sources": manifest, "converter": converter, "source_stats": dict(stats), "splits": dict(split_totals), "vocab_size": len(vocab), "coverage": coverage,
              "source_oov": oov, "source_lengths": lengths, "source_training_coverage": source_training_coverage,
              "source_groups": {source: dict(Counter(groups.values())) for source, groups in frozen_groups.items()},
              "previous_train": previous_manifest,
              "quality_notes": {"naturalconv": "Fixed training-only sample: 12 conversations, 242 utterances, 4503 characters; human review found 13 clear issues (6 near/homophone typos, 2 的/得 errors, 5 expression issues) plus 2 suspected issues, affecting 9 conversations; sports 8/12, frequent greetings, no marketing observed. This sample does not estimate the corpus-wide error rate. Limit to 2M train characters as a small spoken-language supplement; do not correct source text or treat automatic heldout labels as clean human gold. Audit sample: results/retrain-20261021/corpus-naturalconv-training-sample.jsonl, SHA256 e95c8439fc4f6b8bd3cbd2b72b9e31723b787a028421e60f38a92142088d9f82.",
                                "cqia": "Only answer_from=human; exclude xhs marketing source. Keep all distinct human answers to the same namespaced question; deduplicate body text after grouping.",
                                "fresh_evaluation": "Only NaturalConv and CPED are fresh source holdouts; old sources are regression. Automatic pinyin and original spelling can be noisy; never filter by recall/model results."},
              "character_tables": {str(i): digest(sources / f"{i}.txt") for i in (1, 2, 3)},
              "dictionary_sha256": digest(ROOT / "data/dictionaries/8105.dict.yaml"),
              "eval_exclusion_files": exclusion_files, "eval_exclusion_sequences": len(exclusions),
              "rules": {"normalization": "Unicode NFC then OpenCC t2s; preserve Chinese punctuation; dedup canonical applies NFKC; remove segmented-Han spaces, markdown styling, controls; preserve Latin token spaces; join hard wraps between long adjacent prose lines without terminal punctuation",
                        "source_groups": "whole LCCC conversation canonical hash; Wikipedia page id; CQIA source-domain-namespaced original qid or canonical question hash, all distinct answers kept; TNEWS idx; NaturalConv document_id across distinct dialog_id conversations; CPED TV_ID across distinct Dialogue_ID conversations; split before cleaning and paragraph segmentation; frozen new source assignments in source-groups.json",
                        "dialogues": "clean/filter each utterance separately; retain order within one conversation, preserve existing ending punctuation, otherwise append Chinese full stop; concatenate only that conversation, then segment at <=384 characters; CPED sort by Utterance_ID; no cross-conversation concatenation",
                        "split": "BLAKE2b(source NUL split_group) first32bits modulo10000: old sources dev[0,150), holdout[150,300), train otherwise; NaturalConv dev[0,1000), holdout[1000,2000), train otherwise; CPED official train/valid/test files, disjoint TV_ID; hash-shuffled source-stratified character budgets",
                        "observed_source_exclusion": "NaturalConv document_id 0..6 excluded entirely because source research observed them before split freezing",
                        "dedup": "global canonical exact paragraphs >=12 characters; for >=64 characters query >=2 shared keys among12 bottom-hash16-character shingles and verify length ratio>=0.9 plus shared4-character-shingle multiset count divided by larger total>=0.9; digit content retained; training processed before heldouts",
                        "short_common_language": "normalized paragraphs shorter than12 characters retained only in train, at most32 occurrences globally; no short paragraph crosses into heldout",
                        "previous_training_overlap": "Fresh NaturalConv/CPED dev and holdout paragraphs excluded against recorded previous train.txt using the same exact/near canonical paragraph rules, independently of model/candidate results; no previous-training exclusion for new train or old-source regression splits",
                        "limits": limits, "paragraph_max_characters": 384,
                        "filter": "no URL/contact/advertising/code-heavy/low-Han (<45% of letters and digits) paragraphs or >=6 repeated-character runs; navigation headings removed; no medical-vocabulary exclusion",
                        "eval_exclusion": "all public data/eval/*.jsonl plus explicit repeatable --exclude paths: context+gold canonical sequences >=6 characters excluded as substrings from all splits, without model scoring holdout; short common words are not globally forbidden"},
              "script_sha256": digest(Path(__file__)), "outputs": {name: {"sha256": digest(args.output / name), "bytes": (args.output / name).stat().st_size} for name in ("documents.jsonl", "paragraph-map.jsonl", "source-groups.json", "train.txt", "dev.txt", "holdout.txt", "compact-spec.json", "small-spec.json", "large-spec.json")}}
    write_json(args.output / "data-recipe.json", recipe)
    write_json(args.evidence / "data-recipe.json", recipe)
    write_json(args.output / "train-character-frequency.json", frequencies.most_common())
    stage.close()
    seen.close()
    stage_path.unlink()
    (args.output / "dedup.sqlite3").unlink()
    print(json.dumps({"stage": "done", "splits": dict(split_totals), "coverage": coverage, "oov": oov}), flush=True)


if __name__ == "__main__":
    main()
