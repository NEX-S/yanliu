#!/usr/bin/env python3
"""Generate synthetic Chinese cases with deepseek-flash through llmux and check them with yanliu evaluate.

Each round asks the teacher for context/gold/bucket cases, derives the typed input from gold with
the pinned pypinyin and rejects cases outside the fixed rules. A bad teacher batch is recorded and
the next round continues; HTTP, gateway and CLI errors stop the run. One CLI run checks every
accepted case. Validated cases and round records are saved after each round, before the next request.
Exit status: 0 all rounds valid, 2 some rounds failed, 1 no valid case.
--generate-only saves validated cases without observing CLI candidate scores.
--pairs supplies public homophone pairs and requires two nonempty contexts per target.
Round pair_range values are zero-based, with an exclusive end.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

from pypinyin import Style, lazy_pinyin

ROOT = Path(__file__).resolve().parents[1]
TEACHER_MODEL = "deepseek-flash"
PYPINYIN_VERSION = "0.55.0"
BUCKETS = ("daily", "software", "science", "office")
GOLD = re.compile(r"[\u4e00-\u9fff]{2,8}")
CONTEXT = re.compile(r"[\u4e00-\u9fff，。！？；：、（）“”‘’…—]{0,32}")
INPUT = re.compile(r"[a-z]{1,48}")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def request_teacher(endpoint, payload, timeout, output, name):
    """POST one chat completion; the raw response is saved before it is parsed."""
    body = json.dumps(payload, ensure_ascii=False).encode()
    (output / f"{name}-request.json").write_bytes(body)
    request = urllib.request.Request(endpoint + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        error.add_note(error.read().decode(errors="replace"))
        raise
    (output / f"{name}-response.body").write_bytes(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def teacher_cases(response):
    """Return the generated case array; ValueError marks a bad teacher batch."""
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("teacher content must be a JSON string")
    generated = json.loads(content)
    if (not isinstance(generated, dict) or set(generated) != {"cases"}
            or not isinstance(generated["cases"], list)):
        raise ValueError("teacher content must be a JSON object containing only a cases array")
    return generated["cases"]


def pinyin(gold):
    return "".join(lazy_pinyin(gold, style=Style.NORMAL))


def load_pairs(path):
    pairs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs must be a nonempty JSON array")
    seen_inputs = set()
    for index, pair in enumerate(pairs, 1):
        if (not isinstance(pair, list) or len(pair) != 2 or
                not all(isinstance(gold, str) and GOLD.fullmatch(gold) for gold in pair)):
            raise ValueError(f"pair {index}: expected two Chinese targets of 2 to 8 characters")
        if pair[0] == pair[1]:
            raise ValueError(f"pair {index}: duplicate_target")
        typed = pinyin(pair[0])
        if not INPUT.fullmatch(typed) or not INPUT.fullmatch(pinyin(pair[1])):
            raise ValueError(f"pair {index}: pinyin_unavailable_or_length")
        if typed != pinyin(pair[1]):
            raise ValueError(f"pair {index}: pinyin_mismatch")
        if typed in seen_inputs:
            raise ValueError(f"pair {index}: duplicate_pinyin_group")
        seen_inputs.add(typed)
    return pairs


def rejection(case, seen):
    if not isinstance(case, dict) or set(case) != {"context", "gold", "bucket"}:
        return "schema_fields"
    if not all(isinstance(value, str) for value in case.values()):
        return "schema_types"
    if case["bucket"] not in BUCKETS:
        return "unknown_bucket"
    if not GOLD.fullmatch(case["gold"]):
        return "gold_chinese_or_length"
    if not CONTEXT.fullmatch(case["context"]):
        return "context_chinese_or_length"
    if case["gold"] in case["context"]:
        return "answer_in_context"
    if (case["context"], case["gold"]) in seen:
        return "duplicate_case"
    if not INPUT.fullmatch(pinyin(case["gold"])):
        return "pinyin_unavailable_or_length"
    return None


def select_cases(generated, seen, round_number, response_sha256, pairs=None):
    accepted, rejected = [], []
    batch_seen = seen.copy()
    targets = {gold for pair in pairs for gold in pair} if pairs is not None else None
    for index, case in enumerate(generated, 1):
        reason = rejection(case, batch_seen)
        if reason is None and targets is not None and case["gold"] not in targets:
            reason = "unexpected_target"
        if reason is None and pairs is not None and not case["context"]:
            reason = "homophone_empty_context"
        if reason:
            rejected.append({"index": index, "reason": reason, "case": case})
            continue
        batch_seen.add((case["context"], case["gold"]))
        accepted.append((index, {"id": f"teacher-r{round_number:03d}-c{index:03d}-{response_sha256[:12]}",
                                 "input": pinyin(case["gold"]), **case}))
    if pairs is not None:
        groups = {}
        for index, case in accepted:
            groups.setdefault(case["input"], []).append((index, case))
        invalid = set()
        for typed, group in groups.items():
            counts = {}
            for _, case in group:
                counts[case["gold"]] = counts.get(case["gold"], 0) + 1
            reason = ("homophone_missing_pair" if len(counts) < 2 else
                      "homophone_unbalanced" if len(set(counts.values())) != 1 else None)
            if reason is None and set(counts.values()) != {2}:
                reason = "homophone_context_count"
            if reason:
                invalid.add(typed)
                rejected.extend({"index": index, "reason": reason,
                                 "case": {key: case[key] for key in ("context", "gold", "bucket")}}
                                for index, case in group)
        accepted = [(index, case) for index, case in accepted if case["input"] not in invalid]
    accepted = [case for _, case in accepted]
    seen.update((case["context"], case["gold"]) for case in accepted)
    rejected.sort(key=lambda item: item["index"])
    return accepted, rejected


def prompt(round_number, count, seen, focus, pairs=None):
    text = (
        f"请为离线中文拼音输入法生成第{round_number}轮的{count}条全新合成开发案例。"
        "只涉及公开、抽象的日常生活、软件开发、自然科学、办公表达，不涉及真实个人、账号、对话或私有资料。"
        "四个领域尽量均衡，bucket分别为daily、software、science、office。"
        "目标既可以是常用词也可以是自然的短语，不要只选罕见专名。\n"
        "只输出严格JSON对象，唯一顶层字段cases是数组；不要Markdown、代码围栏、解释或推理。"
        "每个案例必须且只能有context、gold、bucket三个字符串字段，不要生成拼音或input字段。"
    )
    if pairs is not None:
        text += ("gold必须严格选自本轮指定的公开同音词对，不得另选词语。"
                 "每对中的每个词各生成两条不同的非空中文前文，总计每对四条案例。"
                 "前文必须能自然紧接目标词，明确排除同对另一个词，不可让两个词都适用。"
                 "指定词对：" + json.dumps(pairs, ensure_ascii=False))
    else:
        text += "gold为连续2至6个简体汉字，不含标点、英文或数字。"
        text += f"前{count // 2}条context必须为空字符串，其余context是适合紧接gold的自然中文前文，"
    text += ("context最多32个汉字或中文标点，不含空白、拉丁字母和数字，且不能已经包含gold。"
             "各案例的目标与前文应多样，不要重复案例。")
    if seen and pairs is None:
        # Bounded so the prompt does not grow with long runs.
        text += "\n以下目标已经生成，请选择新词语：" + "、".join(sorted({gold for _, gold in seen})[-64:])
    if focus:
        text += "\n本次重点：" + focus
    return text


def run(args):
    if version("pypinyin") != PYPINYIN_VERSION:
        raise ValueError(f"pypinyin must be exactly {PYPINYIN_VERSION}")
    endpoint = args.endpoint.rstrip("/")
    pairs = load_pairs(args.pairs) if args.pairs is not None else None
    if pairs is not None and (args.rounds - 1) * (args.cases_per_round // 4) >= len(pairs):
        raise ValueError("--rounds exceeds the available pair batches")
    output, binary, model = args.output_dir.absolute(), args.binary.resolve(), args.model.resolve()
    files = {"generator": sha256(Path(__file__).resolve()), "binary": sha256(binary), "model": sha256(model),
             "dictionaries": {path.name: sha256(path)
                              for path in sorted((ROOT / "data/dictionaries").glob("*.dict.yaml"))}}
    if pairs is not None:
        files["pairs"] = sha256(args.pairs)
    output.mkdir(parents=True)
    data, check = output / "cases.jsonl", output / "check.json"
    seen, case_count, rounds = set(), 0, []
    for number in range(1, args.rounds + 1):
        count, round_pairs, pair_range = args.cases_per_round, None, None
        if pairs is not None:
            start = (number - 1) * (args.cases_per_round // 4)
            round_pairs = pairs[start:start + args.cases_per_round // 4]
            count = len(round_pairs) * 4
            pair_range = {"start": start, "end": start + len(round_pairs)}
        payload = {
            "model": TEACHER_MODEL,
            "messages": [{"role": "user", "content": prompt(number, count, seen, args.focus, round_pairs)}],
            "stream": False, "max_tokens": min(3072, 256 + 192 * count), "temperature": 0.8,
            "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"},
        }
        response, response_sha256 = request_teacher(endpoint, payload, args.request_timeout,
                                                     output, f"round-{number:03d}")
        record = {"round": number, "response_sha256": response_sha256}
        if pair_range is not None:
            record["pair_range"] = pair_range
        try:
            generated = teacher_cases(response)
        except ValueError as error:
            record["error"] = f"{type(error).__name__}: {error}"
        else:
            accepted, rejected = select_cases(generated, seen, number, response_sha256, round_pairs)
            if accepted:
                with data.open("a", encoding="utf-8") as stream:
                    for case in accepted:
                        stream.write(json.dumps(case, ensure_ascii=False) + "\n")
                case_count += len(accepted)
            record.update(generated=len(generated), accepted=[case["id"] for case in accepted], rejected=rejected)
            if not accepted:
                record["error"] = "no valid cases after rule rejection"
        rounds.append(record)
        write_json(output / f"round-{number:03d}.json", record)
        print(json.dumps({"round": number, "generated": record.get("generated"),
                          "accepted": len(record.get("accepted", [])), "error": record.get("error")},
                         ensure_ascii=False), flush=True)
    buckets = {}
    if case_count and not args.generate_only:
        subprocess.run([str(binary), "--model", str(model), "evaluate", "--data", str(data), "--output", str(check)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        report = json.loads(check.read_text(encoding="utf-8"))
        if report["format"] != "yanliu-candidate-check-v1":
            raise ValueError(f"unsupported CLI report format: {report['format']}")
        buckets = report["buckets"]
    failed = [record["round"] for record in rounds if "error" in record]
    status = "failed" if not case_count else "partial" if failed else "complete"
    write_json(output / "summary.json", {
        "status": status, "mode": "pairs" if pairs is not None else "general",
        "generate_only": args.generate_only, "check_skipped": args.generate_only or not case_count,
        "teacher_model": TEACHER_MODEL, "endpoint": endpoint,
        "pypinyin": PYPINYIN_VERSION, "argv": sys.argv, "files": files, "rounds": rounds, "buckets": buckets,
    })
    print(json.dumps({"status": status, "output_dir": str(output), "cases": case_count,
                      "failed_rounds": failed, "all": buckets.get("all")}, ensure_ascii=False), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="new evidence directory; existing destinations are rejected")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/yanliu")
    parser.add_argument("--model", type=Path, default=ROOT / "data/models/sentence-model.safetensors")
    parser.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL"),
                        help="OpenAI-compatible API base including any path prefix; defaults to OPENAI_BASE_URL")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cases-per-round", type=int, default=4)
    parser.add_argument("--pairs", type=Path,
                        help="public JSON homophone pairs; generates two contexts per target, four cases per pair")
    parser.add_argument("--generate-only", action="store_true",
                        help="save validated cases without running CLI evaluation, for unobserved held-out data")
    parser.add_argument("--focus", default="", help="additional content requirements, such as technical terms or longer phrases")
    parser.add_argument("--request-timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or OPENAI_BASE_URL is required")
    if args.rounds < 1 or not 1 <= args.cases_per_round <= 32:
        parser.error("--rounds must be positive and --cases-per-round between 1 and 32")
    if args.pairs is not None and args.cases_per_round % 4:
        parser.error("--pairs requires --cases-per-round divisible by 4")
    return {"complete": 0, "partial": 2, "failed": 1}[run(args)]


if __name__ == "__main__":
    raise SystemExit(main())
