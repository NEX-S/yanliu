"""ime_lab with a fake teacher and fake CLI: no gateway, yanliu binary or personal state."""
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error

SPEC = importlib.util.spec_from_file_location("ime_lab", Path(__file__).resolve().parents[1] / "scripts/ime_lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)
SECRETS = {"OPENAI_API_KEY": "fixture-openai-key", "ANTHROPIC_AUTH_TOKEN": "fixture-anthropic-token"}


def case(**changes):
    return {"context": "", "gold": "排序模型", "bucket": "software", **changes}


def completion(content):
    return {"choices": [{"message": {"content": content}}]}


class ImeLabTests(unittest.TestCase):
    def test_target_pinyin_keeps_phrase_context_but_only_consumes_the_target(self):
        generated = [case(context="银", gold="行", bucket="daily"),
                     case(context="这是一位行", gold="长", bucket="daily")]
        accepted, rejected = lab.select_cases(generated, set(), 1, "a" * 64)
        self.assertEqual(rejected, [])
        self.assertEqual([row["input"] for row in accepted], ["hang", "zhang"])
        self.assertEqual(lab.pinyin("行"), "xing")
        self.assertEqual(lab.pinyin("银行"), "yinhang")

    def test_bad_teacher_batches_are_recorded_and_accepted_cases_are_checked_once(self):
        first = json.dumps({"cases": [case()]})
        last = json.dumps({"cases": [case(gold="银行", bucket="daily")]})
        scenarios = (
            ("complete", [first, json.dumps({"cases": [case(gold="绿色")]}), last], 0, [], 3),
            ("bad-json", [first, '{"cases":[美]}', last], 2, [2], 2),
            ("bad-content-type", [first, None, last], 2, [2], 2),
            ("no-valid-case", [first, first, last], 2, [2], 2),
            ("failed", ['{"cases":[美]}'] * 3, 1, [1, 2, 3], 0),
        )
        for name, contents, code, failed, checked in scenarios:
            with self.subTest(name), tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-") as directory:
                root = Path(directory)
                (root / "data/dictionaries").mkdir(parents=True)
                fixture = root / "fixture"
                fixture.write_bytes(b"binary and model identity only; never executed")
                output = root / "new-parent/evidence"
                requests, evaluated = [], []

                def urlopen(request, timeout):
                    requests.append(request)
                    return io.BytesIO(json.dumps(completion(contents[len(requests) - 1])).encode())

                def cli(command, cwd, check, stdout):
                    self.assertEqual(command[3], "evaluate")
                    data = Path(command[command.index("--data") + 1])
                    evaluated.extend(json.loads(line) for line in data.read_text(encoding="utf-8").splitlines())
                    lab.write_json(Path(command[command.index("--output") + 1]),
                                   {"format": "yanliu-candidate-check-v1", "buckets": {"all": {"n": len(evaluated)}}})

                argv = ["ime_lab.py", "--output-dir", str(output), "--binary", str(fixture), "--model", str(fixture),
                        "--focus", "困难短语"]
                with (patch.object(lab, "ROOT", root), patch.object(lab.sys, "argv", argv),
                      patch.dict(os.environ, {"OPENAI_BASE_URL": "https://gateway.example/openai/", **SECRETS}),
                      patch.object(lab.urllib.request, "urlopen", urlopen),
                      patch.object(lab.subprocess, "run", cli), redirect_stdout(io.StringIO())):
                    self.assertEqual(lab.main(), code)

                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["mode"], "general")
                self.assertFalse(summary["generate_only"])
                self.assertEqual(summary["check_skipped"], not checked)
                self.assertEqual([record["round"] for record in summary["rounds"] if "error" in record], failed)
                self.assertEqual(len(evaluated), checked)
                self.assertEqual(summary["buckets"], {"all": {"n": checked}} if checked else {})
                self.assertEqual(len(requests), 3)
                for record in summary["rounds"]:
                    self.assertEqual(json.loads((output / f"round-{record['round']:03d}.json").read_text()), record)
                for request in requests:
                    self.assertEqual(request.full_url, "https://gateway.example/openai/chat/completions")
                    self.assertIsNone(request.get_header("Authorization"))
                    payload = json.loads(request.data)
                    self.assertEqual(payload["model"], "deepseek-flash")
                    self.assertEqual(payload["thinking"], {"type": "disabled"})
                    self.assertEqual(payload["response_format"], {"type": "json_object"})
                    self.assertIn("前2条context必须为空字符串", payload["messages"][0]["content"])
                self.assertIn("困难短语", json.loads(requests[2].data)["messages"][0]["content"])
                if name == "complete":
                    self.assertIn("排序模型", json.loads(requests[2].data)["messages"][0]["content"])
                    self.assertEqual([row["input"] for row in evaluated], ["paixumoxing", "lvse", "yinhang"])
                if name == "bad-json":
                    raw = json.loads((output / "round-002-response.body").read_text(encoding="utf-8"))
                    self.assertEqual(raw, completion(contents[1]))
                for path in output.iterdir():
                    for secret in SECRETS.values():
                        self.assertNotIn(secret.encode(), path.read_bytes())

    def test_later_http_failure_preserves_completed_round_cases_and_rejections(self):
        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-checkpoint-") as directory:
            root = Path(directory)
            (root / "data/dictionaries").mkdir(parents=True)
            fixture, output = root / "fixture", root / "evidence"
            fixture.write_bytes(b"bound binary and model; not executed")
            response = completion(json.dumps({"cases": [case(), case(gold="模型A")]}))
            error = urllib.error.HTTPError("https://gateway.example/openai/chat/completions", 503,
                                           "fixture unavailable", {}, io.BytesIO(b"fixture unavailable"))
            argv = ["ime_lab.py", "--output-dir", str(output), "--binary", str(fixture), "--model", str(fixture),
                    "--endpoint", "https://gateway.example/openai", "--rounds", "2"]
            with (patch.object(lab, "ROOT", root), patch.object(lab.sys, "argv", argv),
                  patch.object(lab.urllib.request, "urlopen",
                               side_effect=[io.BytesIO(json.dumps(response).encode()), error]) as gateway,
                  patch.object(lab.subprocess, "run") as cli, redirect_stdout(io.StringIO())):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    lab.main()
            self.assertIs(raised.exception, error)
            self.assertEqual(gateway.call_count, 2)
            cli.assert_not_called()
            rows = [json.loads(line) for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [{"id": rows[0]["id"], "input": "paixumoxing", **case()}])
            record = json.loads((output / "round-001.json").read_text(encoding="utf-8"))
            self.assertEqual(record, {
                "round": 1, "response_sha256": lab.sha256(output / "round-001-response.body"), "generated": 2,
                "accepted": [rows[0]["id"]],
                "rejected": [{"index": 2, "reason": "gold_chinese_or_length", "case": case(gold="模型A")}],
            })
            self.assertEqual(json.loads((output / "round-001-response.body").read_text()), response)
            self.assertTrue((output / "round-001-request.json").exists())
            self.assertTrue((output / "round-002-request.json").exists())
            self.assertFalse((output / "round-002.json").exists())
            self.assertFalse((output / "summary.json").exists())
            self.assertFalse((output / "check.json").exists())

    def test_typed_input_and_case_rules(self):
        for gold, typed in (("绿色", "lvse"), ("银行", "yinhang"), ("行走", "xingzou"), ("对", "dui")):
            self.assertEqual(lab.pinyin(gold), typed)
            self.assertIsNone(lab.rejection(case(gold=gold), set()))
        invalid = (
            (case(input="paixumoxing"), "schema_fields"),
            (case(context=None), "schema_types"),
            (case(bucket="private"), "unknown_bucket"),
            (case(gold="模型A"), "gold_chinese_or_length"),
            (case(gold="中" * 9), "gold_chinese_or_length"),
            (case(context="软件Rust"), "context_chinese_or_length"),
            (case(context="中" * 33), "context_chinese_or_length"),
            (case(context="需要排序模型"), "answer_in_context"),
            (case(gold="\u9ffd\u9ffd"), "pinyin_unavailable_or_length"),
        )
        for value, reason in invalid:
            with self.subTest(value=value):
                self.assertEqual(lab.rejection(value, set()), reason)
        seen = set()
        accepted, rejected = lab.select_cases([case(), case(context="现在需要，"), case()], seen, 1, "a" * 64)
        self.assertEqual([row["id"] for row in accepted], [f"teacher-r001-c00{index}-{'a' * 12}" for index in (1, 2)])
        self.assertEqual(accepted[1], {"id": accepted[1]["id"], "input": "paixumoxing", **case(context="现在需要，")})
        self.assertEqual(rejected, [{"index": 3, "reason": "duplicate_case", "case": case()}])
        self.assertEqual(lab.select_cases([case()], seen, 2, "b" * 64)[1][0]["reason"], "duplicate_case")

    def test_teacher_content_must_be_a_cases_object(self):
        self.assertEqual(lab.teacher_cases(completion('{"cases": [null]}')), [None])
        for content in ('```json\n{"cases": []}\n```', "[]", '{"cases": {}}',
                        '{"cases": [], "extra": true}', "", None, [], {}, 0):
            with self.subTest(content=content), self.assertRaises(ValueError):
                lab.teacher_cases(completion(content))

    def test_generate_only_saves_validated_general_cases_without_running_cli(self):
        rows = [case()]
        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-unobserved-") as directory:
            root = Path(directory)
            (root / "data/dictionaries").mkdir(parents=True)
            fixture = root / "fixture"
            fixture.write_bytes(b"bound binary and model; not executed")
            output = root / "evidence"
            response = completion(json.dumps({"cases": rows}))
            argv = ["ime_lab.py", "--output-dir", str(output), "--binary", str(fixture), "--model", str(fixture),
                    "--endpoint", "https://gateway.example/openai", "--rounds", "1", "--cases-per-round", "1",
                    "--generate-only"]
            with (patch.object(lab, "ROOT", root), patch.object(lab.sys, "argv", argv),
                  patch.object(lab.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())),
                  patch.object(lab.subprocess, "run") as cli, redirect_stdout(io.StringIO())):
                self.assertEqual(lab.main(), 0)
            cli.assert_not_called()
            cases = [json.loads(line) for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["input"] for row in cases], ["paixumoxing"])
            self.assertEqual([row["gold"] for row in cases], [row["gold"] for row in rows])
            self.assertFalse((output / "check.json").exists())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["mode"], "general")
            self.assertTrue(summary["generate_only"])
            self.assertTrue(summary["check_skipped"])
            self.assertEqual(summary["buckets"], {})
            self.assertEqual(summary["files"]["binary"], lab.sha256(fixture))
            self.assertEqual(summary["files"]["model"], lab.sha256(fixture))
            self.assertEqual(json.loads((output / "round-001-response.body").read_text()), response)
            self.assertTrue((output / "round-001-request.json").exists())

    def test_pairs_are_validated_with_fixed_pinyin_before_generation(self):
        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-pairs-") as directory:
            path = Path(directory) / "pairs.json"
            pairs = [["意识", "仪式"], ["实际", "时机"]]
            lab.write_json(path, pairs)
            self.assertEqual(lab.load_pairs(path), pairs)
            invalid = (
                ({"pairs": pairs}, "nonempty JSON array"),
                ([], "nonempty JSON array"),
                ([["意识"]], "two Chinese targets"),
                ([["意识", None]], "two Chinese targets"),
                ([["意识", "模型A"]], "two Chinese targets"),
                ([["意识", "意识"]], "duplicate_target"),
                ([["意识", "实验"]], "pinyin_mismatch"),
                ([["\u9ffd\u9ffd", "\u9ffc\u9ffc"]], "pinyin_unavailable_or_length"),
                ([pairs[0], ["医师", "遗失"]], "duplicate_pinyin_group"),
            )
            for value, reason in invalid:
                with self.subTest(value=value):
                    lab.write_json(path, value)
                    with self.assertRaisesRegex(ValueError, reason):
                        lab.load_pairs(path)

    def test_pair_mode_assigns_words_in_order_and_rejects_unrequested_targets(self):
        pairs = [["意识", "仪式"], ["实际", "时机"], ["形式", "行驶"]]
        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-fixed-pairs-") as directory:
            root = Path(directory)
            (root / "data/dictionaries").mkdir(parents=True)
            fixture, source, output = root / "fixture", root / "pairs.json", root / "evidence"
            fixture.write_bytes(b"not executed")
            lab.write_json(source, pairs)
            requests = []

            def urlopen(request, timeout):
                requests.append(json.loads(request.data))
                batch = pairs[:2] if len(requests) == 1 else pairs[2:]
                rows = [case(gold=gold, context=context) for pair in batch for gold in pair
                        for context in ("我们讨论的", "请关注当前")]
                # These two extra targets are a homophone pair but were not requested.
                rows += [case(gold=gold, context="这个测试的") for gold in ("时间", "事件")]
                return io.BytesIO(json.dumps(completion(json.dumps({"cases": rows}))).encode())

            argv = ["ime_lab.py", "--output-dir", str(output), "--binary", str(fixture), "--model", str(fixture),
                    "--endpoint", "https://gateway.example/openai", "--rounds", "2", "--cases-per-round", "8",
                    "--pairs", str(source), "--generate-only"]
            with (patch.object(lab, "ROOT", root), patch.object(lab.sys, "argv", argv),
                  patch.object(lab.urllib.request, "urlopen", urlopen),
                  patch.object(lab.subprocess, "run") as cli, redirect_stdout(io.StringIO())):
                self.assertEqual(lab.main(), 0)
            cli.assert_not_called()
            self.assertEqual(len(requests), 2)
            for payload, batch, count in zip(requests, (pairs[:2], pairs[2:]), (8, 4)):
                text = payload["messages"][0]["content"]
                self.assertIn(f"轮的{count}条", text)
                self.assertIn(json.dumps(batch, ensure_ascii=False), text)
                self.assertIn("每对中的每个词各生成两条", text)
                self.assertIn("明确排除同对另一个词", text)
                self.assertNotIn("开元", text)
                self.assertNotIn(str(source), text)
                self.assertNotIn("以下目标已经生成", text)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "pairs")
            self.assertEqual(summary["files"]["pairs"], lab.sha256(source))
            self.assertEqual([row["pair_range"] for row in summary["rounds"]],
                             [{"start": 0, "end": 2}, {"start": 2, "end": 3}])
            self.assertEqual([item["reason"] for row in summary["rounds"] for item in row["rejected"]],
                             ["unexpected_target"] * 4)
            rows = [json.loads(line) for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["gold"] for row in rows}, {gold for pair in pairs for gold in pair})
            with (patch.object(lab.sys, "argv", [*argv, "--rounds", "3"]),
                  patch.object(lab.urllib.request, "urlopen") as gateway):
                with self.assertRaisesRegex(ValueError, "available pair batches"):
                    lab.main()
            gateway.assert_not_called()

        rows = [case(gold=gold, context="这个测试的") for gold in pairs[0]]
        accepted, rejected = lab.select_cases(rows, set(), 1, "a" * 64, pairs=pairs[:1])
        self.assertEqual(accepted, [])
        self.assertEqual([row["reason"] for row in rejected], ["homophone_context_count"] * 2)
        complete = [case(gold=gold, context=context) for gold in pairs[0]
                    for context in ("我们讨论的", "请关注当前")]
        invalid = (
            (complete[:2], ["homophone_missing_pair"] * 2),
            (complete[:3], ["homophone_unbalanced"] * 3),
            ([{**complete[0], "context": ""}, *complete[1:]],
             ["homophone_empty_context", *["homophone_unbalanced"] * 3]),
        )
        seen = set()
        for rows, reasons in invalid:
            with self.subTest(rows=rows):
                accepted, rejected = lab.select_cases(rows, seen, 1, "a" * 64, pairs=pairs[:1])
                self.assertEqual(accepted, [])
                self.assertEqual([row["reason"] for row in rejected], reasons)
                self.assertEqual(seen, set())
        self.assertEqual(len(lab.select_cases(complete, seen, 2, "b" * 64, pairs=pairs[:1])[0]), 4)

    def test_raw_responses_are_saved_before_parsing_and_http_errors_keep_their_body(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                status, body = {"/ok/chat/completions": (200, b'{"choices": []}'),
                                "/bad/chat/completions": (200, b"broken JSON")}.get(
                                    self.path, (503, b'{"error": "fixture unavailable"}'))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        endpoint = f"http://127.0.0.1:{server.server_port}"
        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-http-") as directory, \
                patch.dict(os.environ, {"no_proxy": "*"}):
            output = Path(directory)
            self.assertEqual(lab.request_teacher(endpoint + "/ok", {}, 5, output, "ok")[0], {"choices": []})
            with self.assertRaises(json.JSONDecodeError):
                lab.request_teacher(endpoint + "/bad", {}, 5, output, "bad")
            self.assertEqual((output / "bad-response.body").read_bytes(), b"broken JSON")
            with self.assertRaises(urllib.error.HTTPError) as raised:
                lab.request_teacher(endpoint + "/down", {}, 5, output, "down")
            self.assertEqual(raised.exception.code, 503)
            self.assertIn("fixture unavailable", raised.exception.__notes__[0])

    def test_continuous_batches_are_checked_and_issues_deduplicated_before_deadline(self):
        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-continuous-") as directory:
            root = Path(directory)
            (root / "data/dictionaries").mkdir(parents=True)
            (root / "data/dictionaries/test.dict.yaml").write_text("public fixture")
            fixture, output = root / "fixture", root / "evidence"
            fixture.write_bytes(b"bound assets; not executed")
            response = completion(json.dumps({"cases": [case(gold="对"), case(gold="如")]}))
            response["usage"] = {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60}
            evaluated = []

            def cli(command, cwd, check, stdout, timeout):
                self.assertGreater(timeout, 0)
                self.assertLessEqual(timeout, 36)
                self.assertEqual(command[1:3], ["--dictionary-dir", str(root / "data/dictionaries")])
                rows = [json.loads(line) for line in Path(command[command.index("--data") + 1]).read_text().splitlines()]
                evaluated.append(rows)
                lab.write_json(Path(command[command.index("--output") + 1]), {
                    "format": "yanliu-candidate-check-v1",
                    "buckets": {"all": {"n": 2, "top1": 0, "top5": 1, "recall": 1,
                                          "static_top1": 1, "static_top5": 1, "rescues": 0, "breakages": 1}},
                    "rows": [{"id": rows[0]["id"], "rank": None, "static_rank": None, "candidates": []},
                             {"id": rows[1]["id"], "rank": 2, "static_rank": 1,
                              "candidates": [{"text": "乳", "consumed": 2, "source": "ExactDictionary", "score_bits": 1},
                                             {"text": "如", "consumed": 2, "source": "ExactDictionary", "score_bits": 2},
                                             {"text": "如此", "consumed": 2, "source": "Completion", "score_bits": 3}]}],
                })

            sleeps = []

            def sleep(seconds):
                # Two complete batches followed by duration expiry during the normal interval.
                self.assertTrue((output / "issues.json").exists())
                self.assertTrue((output / "round-001-check.json").exists())
                sleeps.append(seconds)
                if len(sleeps) == 2:
                    raise lab.DurationElapsed()

            argv = ["ime_lab.py", "--output-dir", str(output), "--binary", str(fixture), "--model", str(fixture),
                    "--dictionary-dir", str(root / "data/dictionaries"), "--endpoint", "https://gateway.example/openai",
                    "--duration-hours", "0.01", "--cases-per-round", "2"]
            with (patch.object(lab, "ROOT", root), patch.object(lab.sys, "argv", argv),
                  patch.object(lab.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as gateway,
                  patch.object(lab.subprocess, "run", cli), patch.object(lab.time, "sleep", sleep),
                  patch.object(lab.signal, "signal"), patch.object(lab.signal, "setitimer"),
                  redirect_stdout(io.StringIO())):
                # Each request needs an independent response stream.
                gateway.side_effect = lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode())
                self.assertEqual(lab.main(), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["completed_rounds"], 2)
            self.assertEqual(summary["cases"], 4)
            self.assertEqual(summary["usage"]["total_tokens"], 120)
            self.assertEqual(summary["buckets"]["all"]["breakages"], 2)
            self.assertEqual(summary["rounds"][0]["issues"], {"new": 3, "repeated": 0, "total": 3})
            self.assertEqual(summary["rounds"][1]["issues"], {"new": 0, "repeated": 3, "total": 3})
            issues = json.loads((output / "issues.json").read_text())["issues"]
            self.assertTrue(all(issue["occurrences"] == 2 and issue["status"] == "suspected" for issue in issues))
            self.assertEqual([c["text"] for c in issues[1]["top_complete_candidates"]], ["乳", "如"])
            self.assertEqual(len((output / "cases.jsonl").read_text().splitlines()), 4)
            self.assertEqual(len(evaluated), 2)

    def test_continuous_http_failure_preserves_first_batch_and_reports_original_failure(self):
        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-continuous-error-") as directory:
            root = Path(directory)
            (root / "data/dictionaries").mkdir(parents=True)
            (root / "data/dictionaries/test.dict.yaml").write_text("public fixture")
            fixture, output = root / "fixture", root / "evidence"
            fixture.write_bytes(b"bound assets; not executed")
            response = completion(json.dumps({"cases": [case(gold="对")]}))
            error = urllib.error.HTTPError("https://gateway.example/openai/chat/completions", 503,
                                          "fixture unavailable", {}, io.BytesIO(b"fixture unavailable"))

            def cli(command, **kwargs):
                rows = [json.loads(line) for line in Path(command[command.index("--data") + 1]).read_text().splitlines()]
                lab.write_json(Path(command[command.index("--output") + 1]), {
                    "format": "yanliu-candidate-check-v1", "buckets": {"all": {"n": 1}},
                    "rows": [{"id": rows[0]["id"], "rank": 1, "static_rank": 1, "candidates": []}],
                })

            argv = ["ime_lab.py", "--output-dir", str(output), "--binary", str(fixture), "--model", str(fixture),
                    "--dictionary-dir", str(root / "data/dictionaries"), "--endpoint", "https://gateway.example/openai",
                    "--duration-hours", "0.01", "--interval-seconds", "0"]
            with (patch.object(lab, "ROOT", root), patch.object(lab.sys, "argv", argv),
                  patch.object(lab.urllib.request, "urlopen", side_effect=[io.BytesIO(json.dumps(response).encode()), error]),
                  patch.object(lab.subprocess, "run", cli), patch.object(lab.signal, "signal"),
                  patch.object(lab.signal, "setitimer"), redirect_stdout(io.StringIO())):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    lab.main()
            self.assertIs(raised.exception, error)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["completed_rounds"], 1)
            self.assertEqual(summary["rounds"][-1]["status"], "failed")
            self.assertIn("503", summary["error"])
            self.assertEqual((output / "round-002-response.body").read_bytes(), b"fixture unavailable")
            self.assertTrue((output / "round-001-check.json").exists())

    def test_continuous_deadline_timeout_reaps_a_waiting_cli(self):
        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-deadline-") as directory:
            root = Path(directory)
            (root / "data/dictionaries").mkdir(parents=True)
            (root / "data/dictionaries/test.dict.yaml").write_text("public fixture")
            fixture, output, pid = root / "fixture", root / "evidence", root / "child.pid"
            fixture.write_text("#!/usr/bin/env python3\nimport os, time\n"
                               f"open({str(pid)!r}, 'w').write(str(os.getpid()))\n"
                               "time.sleep(60)\n")
            fixture.chmod(0o700)
            response = completion(json.dumps({"cases": [case(gold="对")]}))
            argv = ["ime_lab.py", "--output-dir", str(output), "--binary", str(fixture), "--model", str(fixture),
                    "--dictionary-dir", str(root / "data/dictionaries"), "--endpoint", "https://gateway.example/openai",
                    "--duration-hours", "0.0001", "--interval-seconds", "0"]
            with (patch.object(lab, "ROOT", root), patch.object(lab.sys, "argv", argv),
                  patch.object(lab.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(lab.main(), 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertLess(summary["elapsed_seconds"], 2)
            self.assertEqual(summary["completed_rounds"], 0)
            self.assertEqual(summary["rounds"][0]["status"], "interrupted")
            self.assertFalse((output / "round-001-check.json").exists())
            self.assertTrue(pid.exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid.read_text()), 0)

    def test_continuous_http_read_deadline_and_persistence_outside_alarm(self):
        release_body = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()
                self.wfile.flush()
                release_body.wait(2)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(release_body.set)
        write_json = lab.write_json

        def persist(path, value):
            self.assertEqual(lab.signal.getitimer(lab.signal.ITIMER_REAL), (0.0, 0.0))
            write_json(path, value)
            json.loads(path.read_text())

        with tempfile.TemporaryDirectory(prefix="yanliu-ime-lab-http-deadline-") as directory:
            root = Path(directory)
            (root / "data/dictionaries").mkdir(parents=True)
            (root / "data/dictionaries/test.dict.yaml").write_text("public fixture")
            fixture, output = root / "fixture", root / "evidence"
            fixture.write_bytes(b"not executed")
            argv = ["ime_lab.py", "--output-dir", str(output), "--binary", str(fixture), "--model", str(fixture),
                    "--dictionary-dir", str(root / "data/dictionaries"),
                    "--endpoint", f"http://127.0.0.1:{server.server_port}", "--duration-hours", "0.0001"]
            with (patch.object(lab, "ROOT", root), patch.object(lab.sys, "argv", argv),
                  patch.object(lab, "write_json", persist), patch.dict(os.environ, {"no_proxy": "*"}),
                  patch.object(lab.subprocess, "run") as cli, redirect_stdout(io.StringIO())):
                self.assertEqual(lab.main(), 1)
            cli.assert_not_called()
            summary = json.loads((output / "summary.json").read_text())
            self.assertLess(summary["elapsed_seconds"], 2)
            self.assertEqual(summary["completed_rounds"], 0)
            self.assertEqual(summary["rounds"][0]["status"], "interrupted")
            self.assertTrue((output / "round-001-request.json").exists())


if __name__ == "__main__":
    unittest.main()
