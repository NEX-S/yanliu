#!/usr/bin/env python3
"""Prepare the fixed 2026-10-21 public Chinese corpus; never reads local conversations."""
import argparse
from collections import Counter, defaultdict
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
        # Preserve sentence order, split long prose only at punctuation when possible.
        while len(line) > 384:
            cuts = [line.rfind(char, 160, 384) for char in "。！？；.!?;"]
            cut = max(cuts) + 1
            if cut < 160:
                cut = 384
            yield line[:cut]
            line = line[cut:].lstrip()
        if line:
            yield line


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


def source_records(source, paths):
    for path in paths:
        if path.suffix == ".parquet":
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    if source == "lccc":
                        text = "\n\n".join(row["dialog"])
                        yield fingerprint(canonical(text)).hex(), text
                    elif source == "wiki":
                        text = re.split(r"\n\s*(?:参考文献|外部链接|延伸阅读|参考资料|注释|脚注|参考来源|资料来源)[:：]?\s*\n", row["text"], maxsplit=1)[0]
                        yield str(row["id"]), text
                    else:
                        yield str(row["idx"]), row["sentence"]
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
                    group = str(metadata.get("qid") or metadata.get("question_id") or fingerprint(question).hex())
                    # CQIA sometimes synthesizes instructions around human encyclopedia prose.
                    yield group, row["output"]


def evaluation_exclusions(paths):
    values = set()
    files = []
    for path in paths:
        files.append({"path": str(path), "sha256": digest(path)})
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                golds, context = row.get("golds", [row.get("gold")]), row.get("context", "")
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
    for name, layers, width, heads in (("small", 6, 256, 4), ("large", 8, 448, 8)):
        write_json(output / f"{name}-spec.json", {"config": {"vocab": len(vocab), "n_layer": layers,
                   "n_head": heads, "n_embd": width, "context": 64, "dropout": 0.1},
                   "vocab": vocab, "provenance": provenance})
    return vocab, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/cache/general-model-20261021")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--lccc-chars", type=int, default=35_000_000)
    parser.add_argument("--wiki-chars", type=int, default=35_000_000)
    parser.add_argument("--cqia-chars", type=int, default=12_000_000)
    parser.add_argument("--tnews-chars", type=int, default=2_000_000)
    args = parser.parse_args()
    converter = {"package": "OpenCC", "version": importlib.metadata.version("OpenCC"),
                 "module": opencc.__file__, "configuration": "t2s"}
    args.output.mkdir(parents=True, exist_ok=False)
    args.evidence.mkdir(parents=True, exist_ok=True)
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
                             "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}", "revision": revision,
                             **ATTRIBUTIONS[source]})
    exclusions, exclusion_files = evaluation_exclusions(sorted((ROOT / "data/eval").glob("*.jsonl")))
    exclusion_pattern = re.compile("|".join(re.escape(value) for value in sorted(exclusions))) if exclusions else None
    stats = defaultdict(Counter)
    stage_path = args.output / "staging.sqlite3"
    stage = sqlite3.connect(stage_path)
    stage.execute("PRAGMA journal_mode=OFF")
    stage.execute("PRAGMA synchronous=OFF")
    stage.execute("CREATE TABLE docs(source TEXT, group_id TEXT, split TEXT, shuffle BLOB, paragraphs TEXT, PRIMARY KEY(source,group_id)) WITHOUT ROWID")
    for source, files in SOURCE_FILES.items():
        for group, text in source_records(source, [sources / item[0] for item in files]):
            stats[source]["raw_groups"] += 1
            paragraphs = list(clean_paragraphs(text))
            if not paragraphs:
                stats[source]["empty_after_cleaning"] += 1
                continue
            split = group_split(source, group)
            before = stage.total_changes
            stage.execute("INSERT OR IGNORE INTO docs VALUES(?,?,?,?,?)", (source, group, split, fingerprint(group), json.dumps(paragraphs, ensure_ascii=False)))
            if before == stage.total_changes:
                stats[source]["repeated_group"] += 1
            if stats[source]["raw_groups"] % 50000 == 0:
                stage.commit()
                print(json.dumps({"stage": "clean", "source": source, "raw_groups": stats[source]["raw_groups"]}), flush=True)
        stage.commit()
        print(json.dumps({"stage": "clean_done", "source": source, "stats": stats[source]}), flush=True)
    stage.execute("CREATE INDEX split_order ON docs(split,shuffle)")
    seen = sqlite3.connect(args.output / "dedup.sqlite3")
    seen.execute("PRAGMA journal_mode=OFF")
    seen.execute("PRAGMA synchronous=OFF")
    seen.execute("CREATE TABLE fingerprints(key BLOB PRIMARY KEY) WITHOUT ROWID")
    seen.execute("CREATE TABLE texts(id INTEGER PRIMARY KEY, text TEXT)")
    seen.execute("CREATE TABLE shingles(key BLOB, document INTEGER, PRIMARY KEY(key,document)) WITHOUT ROWID")
    short_counts = Counter()
    frequencies = Counter()
    source_frequencies = defaultdict(Counter)
    source_lengths = defaultdict(Counter)
    split_totals = defaultdict(Counter)
    source_budgets = {(source, split): int(limit * (0.97 if split == "train" else 0.015)) for source, limit in limits.items() for split in ("train", "dev", "holdout")}
    docs_path = args.output / "documents.jsonl"
    with docs_path.open("x", encoding="utf-8") as docs, \
            (args.output / "train.txt").open("x", encoding="utf-8") as train, \
            (args.output / "dev.txt").open("x", encoding="utf-8") as dev, \
            (args.output / "holdout.txt").open("x", encoding="utf-8") as holdout, \
            (args.output / "paragraph-map.jsonl").open("x", encoding="utf-8") as mapping:
        outputs = {"train": train, "dev": dev, "holdout": holdout}
        # Training first prevents a held-out copy of a training paragraph from surviving.
        for split in ("train", "dev", "holdout"):
            for source, group, raw in stage.execute("SELECT source,group_id,paragraphs FROM docs WHERE split=? ORDER BY shuffle", (split,)):
                key = (source, split)
                if stats[source][split + "_characters"] >= source_budgets[key]:
                    continue
                accepted = []
                for text in json.loads(raw):
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
                        exact = fingerprint(normalized)
                        if seen.execute("SELECT 1 FROM fingerprints WHERE key=?", (exact,)).fetchone():
                            stats[source]["exact_duplicate_removed"] += 1
                            continue
                        keys = near_keys(normalized)
                        if keys:
                            placeholders = ",".join("?" for _ in keys)
                            matches = seen.execute(f"SELECT document FROM shingles WHERE key IN ({placeholders}) GROUP BY document HAVING COUNT(*)>=2", keys).fetchall()
                            if any(copied_text(normalized, seen.execute("SELECT text FROM texts WHERE id=?", (row[0],)).fetchone()[0]) for row in matches):
                                stats[source]["near_duplicate_removed"] += 1
                                continue
                            document = seen.execute("INSERT INTO texts(text) VALUES(?)", (normalized,)).lastrowid
                            seen.executemany("INSERT INTO shingles VALUES(?,?)", ((value, document) for value in keys))
                        seen.execute("INSERT INTO fingerprints VALUES(?)", (exact,))
                    accepted.append(text)
                if not accepted:
                    continue
                stats[source][split + "_groups"] += 1
                for number, text in enumerate(accepted):
                    row = {"source": source, "group": group, "paragraph": number, "split": split, "text": text}
                    docs.write(json.dumps(row, ensure_ascii=False) + "\n")
                    outputs[split].write(text + "\n")
                    split_totals[split]["paragraphs"] += 1
                    split_totals[split]["characters"] += len(text)
                    mapping.write(json.dumps({"split": split, "line": split_totals[split]["paragraphs"], "source": source, "group": group, "paragraph": number}, ensure_ascii=False) + "\n")
                    stats[source][split + "_characters"] += len(text)
                    stats[source][split + "_paragraphs"] += 1
                    source_frequencies[source].update(text)
                    source_lengths[source][len(text)] += 1
                    if split == "train":
                        frequencies.update(text)
                if split_totals[split]["paragraphs"] % 50000 < len(accepted):
                    seen.commit()
                    print(json.dumps({"stage": "select", "split": split, "totals": split_totals[split]}), flush=True)
            seen.commit()
    vocab, coverage = make_specs(args.output, sources, ROOT / "data/dictionaries/8105.dict.yaml", frequencies,
                                "LCCC-base (THU-CoAI), Wikimedia Chinese Wikipedia 20231101, COIG-CQIA human answers, CLUE TNEWS; fixed revisions and hashes in data-recipe.json")
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
    recipe = {"sources": manifest, "converter": converter, "source_stats": dict(stats), "splits": dict(split_totals), "vocab_size": len(vocab), "coverage": coverage,
              "source_oov": oov, "source_lengths": lengths, "character_tables": {str(i): digest(sources / f"{i}.txt") for i in (1, 2, 3)},
              "dictionary_sha256": digest(ROOT / "data/dictionaries/8105.dict.yaml"),
              "eval_exclusion_files": exclusion_files, "eval_exclusion_sequences": len(exclusions),
              "rules": {"normalization": "Unicode NFC then OpenCC t2s; preserve Chinese punctuation; dedup canonical applies NFKC; remove segmented-Han spaces, markdown styling, controls; preserve Latin token spaces; join hard wraps between long adjacent prose lines without terminal punctuation",
                        "source_groups": "whole LCCC conversation canonical hash; Wikipedia page id; CQIA original qid or canonical question hash; TNEWS idx; split before paragraph segmentation",
                        "split": "BLAKE2b(source NUL group) first32bits modulo10000: dev[0,150), holdout[150,300), train otherwise; hash-shuffled source-stratified character budgets",
                        "dedup": "global canonical exact paragraphs >=12 characters; for >=64 characters query >=2 shared keys among12 bottom-hash16-character shingles and verify length ratio>=0.9 plus shared4-character-shingle multiset count divided by larger total>=0.9; digit content retained; training processed before heldouts",
                        "short_common_language": "normalized paragraphs shorter than12 characters retained only in train, at most32 occurrences globally; no short paragraph crosses into heldout",
                        "limits": limits, "paragraph_max_characters": 384,
                        "filter": "no URL/contact/advertising/code-heavy/low-Han (<45% of letters and digits) paragraphs or >=6 repeated-character runs; navigation headings removed; no medical-vocabulary exclusion",
                        "eval_exclusion": "all public data/eval/*.jsonl context+gold canonical sequences >=6 characters excluded as substrings, without model scoring holdout"},
              "script_sha256": digest(Path(__file__)), "outputs": {name: {"sha256": digest(args.output / name), "bytes": (args.output / name).stat().st_size} for name in ("documents.jsonl", "paragraph-map.jsonl", "train.txt", "dev.txt", "holdout.txt", "small-spec.json", "large-spec.json")}}
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
