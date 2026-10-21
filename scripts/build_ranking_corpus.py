#!/usr/bin/env python3
"""Build fixed public Chinese ranking cases without observing decoder/model results.

Input is prepare_general_corpus.py's documents.jsonl, with each document's
paragraphs contiguous. Whole-document train/dev/holdout assignments are retained.
Automatic whole-run pypinyin annotation can still misread polyphonic characters;
these natural-text cases are automatic supervision, not human pronunciation gold.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
from importlib.metadata import version
import json
from pathlib import Path
import re

from pypinyin import Style, lazy_pinyin

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "holdout")
TRAIN_LIMITS = {"lccc": 25_000, "wiki": 25_000, "cqia": 8_000, "tnews": 2_000}
EVAL_LIMITS = {"lccc": 750, "wiki": 750, "cqia": 240, "tnews": 60}
HAN = re.compile(r"[\u3400-\u9fff\U00020000-\U000323af]+")
SYLLABLE = re.compile(r"[a-z]+")
LENGTHS = ((1, 1), (2, 2), (3, 4), (5, 8))
COUNT_FIELDS = ("documents", "paragraphs", "han_runs", "proposed_slots",
                "run_too_short_for_selected_length_and_context", "fitting_proposals",
                "document_slot_not_selected", "selected_document_slots", "pypinyin_calls",
                "hash_pool_not_selected", "annotation_pool_cases",
                "non_ascii_full_pinyin", "sampled_document_cases", "existing_evaluation_query",
                "duplicate_exact_query", "unique_eligible_cases", "written", "source_budget_not_selected")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashed(*parts):
    value = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).digest()


def query_key(case, gold=None):
    return hashed(case["input"], case.get("context", ""), case["gold"] if gold is None else gold)


def read_documents(path):
    """Validate the concrete grouped paragraph stream written by the corpus builder."""
    seen, paragraphs = set(), []
    current = None
    paragraph_ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if (not isinstance(row, dict)
                    or set(row) != {"source", "group", "paragraph", "split", "text"}
                    or not isinstance(row["source"], str) or row["source"] not in TRAIN_LIMITS
                    or not isinstance(row["group"], str) or not row["group"]
                    or not isinstance(row["split"], str) or row["split"] not in SPLITS
                    or type(row["paragraph"]) is not int or row["paragraph"] < 0
                    or not isinstance(row["text"], str) or not row["text"]):
                raise ValueError(f"{path}:{number}: invalid public document paragraph")
            key = (row["source"], row["group"])
            if key != current:
                group_hash = hashed(*key)
                if group_hash in seen:
                    raise ValueError(f"{path}:{number}: document group is repeated or crosses split")
                if paragraphs:
                    yield paragraphs
                seen.add(group_hash)
                current, paragraphs, paragraph_ids = key, [], set()
            if paragraphs and row["split"] != paragraphs[0]["split"]:
                raise ValueError(f"{path}:{number}: document group crosses split")
            if row["paragraph"] in paragraph_ids:
                raise ValueError(f"{path}:{number}: duplicate document paragraph index")
            paragraph_ids.add(row["paragraph"])
            paragraphs.append(row)
    if paragraphs:
        yield paragraphs


def document_proposals(paragraphs, counts):
    first = paragraphs[0]
    source_group = first["source"] + ":" + first["group"]
    slots = range(4) if first["split"] == "train" else [hashed(source_group, "length-slot")[0] % 4]
    best = {}
    fitting = 0
    counts["documents"] += 1
    counts["paragraphs"] += len(paragraphs)
    for paragraph in paragraphs:
        for run in HAN.finditer(paragraph["text"]):
            text = run.group()
            counts["han_runs"] += 1
            for slot in slots:
                counts["proposed_slots"] += 1
                identity = (source_group, paragraph["paragraph"], run.start(), slot)
                selection = hashed(*identity, "cut")
                low, high = LENGTHS[slot]
                length = low + selection[0] % (high - low + 1)
                contextual = hashed(source_group, slot, "context")[0] % 4 != 0
                minimum = int(contextual)
                if len(text) < length + minimum:
                    counts["run_too_short_for_selected_length_and_context"] += 1
                    continue
                start = minimum + int.from_bytes(selection[1:9], "big") % (len(text) - length - minimum + 1)
                priority = hashed(*identity, "document-slot")
                fitting += 1
                if slot not in best or priority < best[slot][0]:
                    best[slot] = (priority, identity, text, start, length, contextual)
    counts["fitting_proposals"] += fitting
    counts["document_slot_not_selected"] += fitting - len(best)
    counts["selected_document_slots"] += len(best)
    proposals = []
    for slot in sorted(best):
        _, identity, text, start, length, contextual = best[slot]
        low, high = LENGTHS[slot]
        case = {"id": "public-" + hashed(*identity).hex()[:24],
                "context": text[max(0, start - 32):start] if contextual else "",
                "gold": text[start:start + length],
                "bucket": f"public-{first['source']}-len{low}-{high}-{'context' if contextual else 'no-context'}",
                "source_group": source_group, "group_id": source_group}
        proposals.append({"case": case, "run": text, "target_start": start})
    counts["sampled_document_cases"] += len(proposals)
    return proposals


def annotate(proposal):
    text = proposal["run"]
    syllables = lazy_pinyin(text, style=Style.NORMAL, strict=True, errors=lambda value: list(value))
    if len(syllables) != len(text):
        raise ValueError("pypinyin did not preserve whole-Han-run character alignment")
    case = proposal["case"]
    start = proposal["target_start"]
    target_syllables = syllables[start:start + len(case["gold"])]
    if any(SYLLABLE.fullmatch(value) is None for value in target_syllables):
        return None
    return dict(case, input="".join(target_syllables))


def evaluation_queries(paths):
    queries, files = set(), []
    for path in paths:
        path = Path(path)
        files.append({"path": str(path.resolve()), "sha256": digest(path)})
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("input"), str):
                    continue
                golds = row.get("gold", row.get("golds", []))
                if isinstance(golds, str):
                    golds = [golds]
                if not isinstance(row.get("context", ""), str) or not isinstance(golds, list):
                    raise ValueError(f"{path}: invalid evaluation query")
                for gold in golds:
                    if not isinstance(gold, str):
                        raise ValueError(f"{path}: invalid evaluation gold")
                    queries.add(query_key(row, gold))
    return queries, files


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def build(documents, output, evidence, exclusion_paths,
          train_limits=TRAIN_LIMITS, eval_limits=EVAL_LIMITS):
    documents, output, evidence = map(Path, (documents, output, evidence))
    for path in ([output / f"ranking-{split}.jsonl" for split in SPLITS]
                 + [evidence / f"ranking-data-{name}.json" for name in ("recipe", "summary")]):
        if path.exists():
            raise FileExistsError(path)
    document_sha = digest(documents)
    excluded, exclusion_files = evaluation_queries(exclusion_paths)
    selected = defaultdict(list)
    counts = {(split, source): Counter({name: 0 for name in COUNT_FIELDS})
              for split in SPLITS for source in TRAIN_LIMITS}
    for paragraphs in read_documents(documents):
        first = paragraphs[0]
        key = (first["split"], first["source"])
        stats = counts[key]
        limit = (train_limits if key[0] == "train" else eval_limits)[key[1]]
        if type(limit) is not int or limit < 1:
            raise ValueError("per-source case limits must be positive integers")
        for proposal in document_proposals(paragraphs, stats):
            case = proposal["case"]
            priority = int.from_bytes(hashed(case["id"], "source-budget"), "big")
            entry = (-priority, case["id"], proposal)
            heap = selected[key]
            pool_limit = (limit * 5 + 3) // 4
            if len(heap) < pool_limit:
                heapq.heappush(heap, entry)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    if digest(documents) != document_sha:
        raise ValueError("documents changed during ranking-case construction")
    output.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    summaries, seen = {}, set()
    for split in SPLITS:
        cases = []
        for source in TRAIN_LIMITS:
            stats = counts[(split, source)]
            proposals = sorted(selected[(split, source)], key=lambda entry: (-entry[0], entry[1]))
            stats["annotation_pool_cases"] = len(proposals)
            stats["hash_pool_not_selected"] = stats["sampled_document_cases"] - len(proposals)
            limit = (train_limits if split == "train" else eval_limits)[source]
            for _, _, proposal in proposals:
                stats["pypinyin_calls"] += 1
                case = annotate(proposal)
                if case is None:
                    stats["non_ascii_full_pinyin"] += 1
                    continue
                identity = query_key(case)
                if identity in excluded:
                    stats["existing_evaluation_query"] += 1
                    continue
                if identity in seen:
                    stats["duplicate_exact_query"] += 1
                    continue
                seen.add(identity)
                stats["unique_eligible_cases"] += 1
                if stats["written"] < limit:
                    cases.append(case)
                    stats["written"] += 1
            stats["source_budget_not_selected"] = stats["unique_eligible_cases"] - stats["written"]
        path = output / f"ranking-{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for case in cases:
                stream.write(json.dumps(case, ensure_ascii=False) + "\n")
        summaries[split] = {"path": str(path.resolve()), "sha256": digest(path), "cases": len(cases),
                            "source": dict(Counter(case["source_group"].split(":", 1)[0] for case in cases)),
                            "target_length": dict(sorted(Counter(len(case["gold"]) for case in cases).items())),
                            "context": dict(Counter("present" if case["context"] else "empty" for case in cases)),
                            "source_stats": {source: dict(counts[(split, source)]) for source in TRAIN_LIMITS}}
    recipe = {"format": "yanliu-public-ranking-data-v1", "documents": {"path": str(documents.resolve()), "sha256": document_sha},
              "script_sha256": digest(Path(__file__)), "pypinyin_version": version("pypinyin"),
              "exclusion_files": exclusion_files, "existing_evaluation_queries": len(excluded),
              "rules": {"split": "inherit each complete source/group document assignment; never reassign cases",
                        "document_caps": {"train": 4, "dev": 1, "holdout": 1},
                        "lengths": "one train slot each for 1,2,3-4,5-8 characters; dev/holdout slot chosen by SHA256",
                        "sampling": "SHA256(document, paragraph, Han-run offset, slot) fixes cut and length; minimum document-slot hash chooses paragraph/run; independent source-budget hash selects a fixed pool of ceil(1.25 * target) before any pypinyin call; annotate/filter/deduplicate only this pool, retain at most target in original hash order; no adaptive resampling to fill shortfalls",
                        "context": "document-slot hash requests empty context 25% of the time; otherwise preserve 1..32 preceding Han characters from the same run",
                        "pinyin": "pypinyin NORMAL strict on each complete contiguous Han run before target slicing; one lowercase ASCII syllable per target character; automatic polyphonic readings are not human gold",
                        "dedup": "SHA256 of exact (input,context,gold), against recorded existing evaluation and globally across the fixed annotation pools; different contexts of a common word remain eligible",
                        "selection": "public text only; no CLI calls, model scores, candidate filtering or gold injection into recall",
                        "train_limits": train_limits, "dev_and_holdout_limits": eval_limits,
                        "annotation_pool_limits": {split: {source: (limit * 5 + 3) // 4 for source, limit in
                            (train_limits if split == "train" else eval_limits).items()} for split in SPLITS}},
              "splits": summaries}
    write_json(evidence / "ranking-data-recipe.json", recipe)
    write_json(evidence / "ranking-data-summary.json", summaries)
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=ROOT / "data/cache/general-model-20261021/documents.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cache/general-model-20261021")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "results/general-model-20261021")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="additional evaluation JSONL to exclude; may be repeated")
    args = parser.parse_args()
    exclusions = sorted((ROOT / "data/eval").glob("*.jsonl")) + args.exclude
    recipe = build(args.documents, args.output_dir, args.evidence_dir, exclusions)
    print(json.dumps({split: value["cases"] for split, value in recipe["splits"].items()}), flush=True)


if __name__ == "__main__":
    main()
