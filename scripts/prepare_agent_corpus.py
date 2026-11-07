#!/usr/bin/env python3
import argparse
from collections import Counter
from compression import zstd
import hashlib
import json
from pathlib import Path
import re
import unicodedata

HAN = re.compile(r"[\u3400-\u9fff\U00020000-\U000323af]")
LINK = re.compile(r"!?\[([^\]]+)\]\([^)]*\)")
PRIVATE = re.compile(
    r"https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"(?:^|[\s(（\"'])(?:/|~/|[A-Za-z]:\\)\S+|"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_./-]+|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|"
    r"(?<!\d)1[3-9]\d{9}(?!\d)|"
    r"(?:api[_ -]?key|secret|password|passwd|authorization|bearer|token|密码|密钥|令牌)\s*[:=：]|"
    r"[A-Za-z0-9_+/=-]{32,}", re.I)
COMMAND = re.compile(
    r"^(?:\$\s*|(?:sudo|env|git|cargo|rustc|python[0-9.]*|pip[0-9.]*|node|npm|pnpm|"
    r"yarn|uv|bash|sh|zsh|cd|ls|cat|rg|grep|sed|awk|curl|wget|make|cmake|ctest|docker|"
    r"podman|kubectl|systemctl|journalctl|ssh|scp|rsync|chmod|chown|mkdir|rm|mv|cp)\s+[A-Za-z0-9_.-])")


def paragraphs(text):
    fenced = False
    for line in unicodedata.normalize("NFKC", text).splitlines():
        if re.match(r"^\s*(?:```|~~~)", line):
            fenced = not fenced
            continue
        if fenced or line.startswith("    ") or line.count("|") >= 2:
            continue
        line = LINK.sub(r"\1", line)
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r"^\s*(?:#{1,6}\s*|[-*+>]\s+|\d+[.、)]\s*)", "", line)
        line = line.replace("**", "").replace("__", "").strip()
        if PRIVATE.search(line) or re.search(r"[{}<>\\]", line):
            continue
        han = len(HAN.findall(line))
        if han < 12 or han / max(1, sum(c.isalnum() for c in line)) < 0.55:
            continue
        while len(line) > 384:
            cut = max(line.rfind(c, 160, 384) for c in "。！？；") + 1
            if cut < 160:
                cut = 384
            yield line[:cut]
            line = line[cut:].lstrip()
        if line:
            yield line


def read_records(path, size, stats):
    if path.suffix == ".zst":
        with zstd.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)
        return
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            line = stream.readline(remaining)
            remaining -= len(line)
            if not line:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if remaining or line.endswith(b"\n"):
                    raise
                stats["incomplete_tail"] += 1


def assistant_text(source, record, group):
    if source == "claude":
        message = record.get("message", {})
        if record.get("type") != "assistant" or message.get("role") != "assistant":
            return group, ()
        group = record.get("sessionId") or group
        content = message["content"]
        return group, (content,) if isinstance(content, str) else tuple(
            item["text"] for item in content if item.get("type") == "text")
    if source == "grok":
        return group, (record["content"],) if record.get("type") == "assistant" else ()
    payload = record.get("payload", {})
    if record.get("type") == "session_meta":
        return payload["id"], ()
    if (record.get("type") != "response_item" or payload.get("type") != "message"
            or payload.get("role") != "assistant" or payload.get("channel") == "analysis"):
        return group, ()
    return group, tuple(item["text"] for item in payload["content"]
                        if item.get("type") == "output_text")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    home = Path.home()
    roots = {"claude": home / ".claude/projects", "codex": home / ".codex/sessions",
             "grok": home / ".grok/sessions"}
    seen = set()
    stats = {source: Counter() for source in roots}
    with (args.output / "documents.jsonl").open("w") as output, (args.output / "sources.jsonl").open("w") as sources:
        for source, root in roots.items():
            if not root.is_dir():
                raise FileNotFoundError(root)
            paths = sorted(root.rglob("chat_history.jsonl")) if source == "grok" else sorted(
                p for p in root.rglob("*") if p.name.endswith((".jsonl", ".jsonl.zst")))
            if not paths:
                raise FileNotFoundError(f"No session files in {root}")
            for path in paths:
                size = path.stat().st_size
                digest = hashlib.sha256()
                with path.open("rb") as raw:
                    remaining = size
                    while remaining:
                        block = raw.read(min(1024 * 1024, remaining))
                        digest.update(block)
                        remaining -= len(block)
                group = path.parent.name if source == "grok" else path.name.removesuffix(".zst").removesuffix(".jsonl")
                count = 0
                for record in read_records(path, size, stats[source]):
                    group, texts = assistant_text(source, record, group)
                    for text in texts:
                        stats[source]["assistant_blocks"] += 1
                        for text in paragraphs(text):
                            key = hashlib.sha256(text.encode()).digest()
                            if key in seen:
                                stats[source]["duplicate_paragraphs"] += 1
                                continue
                            seen.add(key)
                            if COMMAND.search(text):
                                stats[source]["command_paragraphs_removed"] += 1
                                continue
                            row = {"source": "local-" + source, "group": group, "text": text,
                                   "source_file": str(path)}
                            output.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stats[source]["paragraphs"] += 1
                            stats[source]["characters"] += len(text)
                            count += 1
                sources.write(json.dumps({"source": source, "path": str(path), "bytes": size,
                                          "sha256": digest.hexdigest(), "paragraphs": count}) + "\n")
                stats[source]["files"] += 1
            print(json.dumps({"source": source, **stats[source]}), flush=True)
    summary = {"private": True, "sources": stats, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "documents_sha256": hashlib.sha256((args.output / "documents.jsonl").read_bytes()).hexdigest(),
               "filter": "assistant text only; no tools/reasoning/user text; source lines with at least 12 Han characters and 55% Han/alphanumeric, split into at most 384 characters; strip code blocks and inline code; exclude paths, URLs, email, IP addresses, phone numbers, command prefixes and credential patterns; exact paragraph deduplication"}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
