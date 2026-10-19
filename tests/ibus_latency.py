#!/usr/bin/python3
"""Private native IBus round-trip and eventual model ordering measurement.

No claims about quality or GNOME Shell frame latency. References are synchronous
CLI outputs from the same assets; only actual changed-order cases time a model result.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import signal
import shlex
from xml.sax.saxutils import escape
import subprocess
import tempfile
import time
import gi

gi.require_version("IBus", "1.0")
from gi.repository import Gio, GLib, IBus
from ibus_panel import panel_state, assert_inline_hidden

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "target/release/yanliu-ibus")
parser.add_argument("--reference", type=pathlib.Path, default=ROOT / "target/release/yanliu")
parser.add_argument("--data-dir", type=pathlib.Path, default=ROOT / "data")
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
if args.repeats < 1:
    parser.error("--repeats must be positive")
args.binary, args.reference, args.data_dir = args.binary.resolve(), args.reference.resolve(), args.data_dir.resolve()
weights = args.data_dir / "models/sentence-model.safetensors"
inputs = ["nihao", "tabashufangzaizhuozishang", "zheshiyigehaozhuyi", "qiurilizhengzuoshandouhongle", "womingnianxiangxuehanyu"]



signal_errors = []

def drain():
    while GLib.MainContext.default().iteration(False):
        pass
    if signal_errors:
        raise signal_errors[0]


def wait(predicate, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        drain()
        if predicate():
            return
        time.sleep(0.0002)
    raise AssertionError("Timed out waiting for private IBus measurement")


def reference(text, model):
    command = [str(args.reference), "--dictionary-dir", str(args.data_dir / "dictionaries")]
    if model:
        command += ["--model", str(weights)]
    command += ["decode", text]
    return [line.split("\t")[1] for line in subprocess.check_output(command, text=True).splitlines()]


def summary(values):
    values = sorted(values)
    return {"n": len(values), **{f"p{int(p*100)}_us": values[max(0, math.ceil(len(values)*p)-1)] for p in [0.5, 0.95, 0.99]}, "max_us": values[-1]}


# Reference startup/loading is outside every timing interval.
cases = [{"input": text, "dictionary": reference(text, False), "expected": reference(text, True)} for text in inputs]
assert any(case["dictionary"] != case["expected"] for case in cases), "measurement must include an actual model promotion"
key_samples, final_samples = [], []
with tempfile.TemporaryDirectory(prefix="yanliu-latency-") as directory:
    directory = pathlib.Path(directory)
    prefix = directory / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "share/ibus/component").mkdir(parents=True)
    (prefix / "bin/yanliu-ibus").symlink_to(args.binary)
    (prefix / "share/yanliu").symlink_to(args.data_dir, target_is_directory=True)
    config_file = directory / "config/yanliu/config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"data_dir": str(args.data_dir), "model": str(weights)}))
    (prefix / "share/ibus/component/yanliu.xml").write_text(
        (ROOT / "packaging/yanliu.xml.in").read_text().replace("@PREFIX@", str(prefix))
        .replace("@CONFIG@", escape(shlex.quote(str(config_file)))))
    address = "unix:path=" + str(directory / "bus")
    env = dict(os.environ, IBUS_ADDRESS=address, IBUS_COMPONENT_PATH=str(prefix / "share/ibus/component"),
               XDG_STATE_HOME=str(directory / "state"), XDG_CONFIG_HOME=str(directory / "config"), XDG_CACHE_HOME=str(directory / "cache"), XDG_DATA_HOME=str(directory / "data"))
    log = open(directory / "daemon.log", "w+")
    daemon = subprocess.Popen(["ibus-daemon", "--single", "--panel=disable", "--config=disable", "--emoji-extension=disable", "--cache=none", "--address=" + address], env=env, stdout=log, stderr=log)
    service_pid = None
    try:
        wait(lambda: (directory / "bus").exists())
        connection = Gio.DBusConnection.new_for_address_sync(address, Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION, None, None)
        def call(path, interface, method, parameters=None):
            return connection.call_sync("org.freedesktop.IBus", path, interface, method, parameters, None, Gio.DBusCallFlags.NONE, 15000, None)
        path = call("/org/freedesktop/IBus", "org.freedesktop.IBus", "CreateInputContext", GLib.Variant("(s)", ("yanliu-latency",))).unpack()[0]
        interface = "org.freedesktop.IBus.InputContext"
        state = {"input": None, "lookups": [], "complete_ns": None, "target": None, "first_ns": {}}
        def received(_c, _s, _p, _i, name, parameters, _d):
            try:
                if name == "UpdatePreeditText":
                    assert_inline_hidden(parameters)
                    state["input"] = None
                elif name == "UpdateAuxiliaryText":
                    raw, cursor, visible = panel_state(parameters)
                    assert cursor == len(raw)
                    state["input"] = raw
                elif name == "UpdateLookupTable":
                    assert state["input"] is not None, "lookup omitted this publish's auxiliary"
                    table = IBus.Serializable.deserialize_object(parameters.get_child_value(0).get_variant())
                    texts = [table.get_candidate(i).get_text() for i in range(table.get_number_of_candidates())]
                    visible = parameters.get_child_value(1).get_boolean()
                    state["lookups"].append((state["input"], texts, visible))
                    if visible:
                        state["first_ns"].setdefault(state["input"], time.perf_counter_ns())
                    target = state["target"]
                    if target and visible and state["input"] == target["input"] and texts == target["expected"] and state["complete_ns"] is None:
                        state["complete_ns"] = time.perf_counter_ns()
                if name == "UpdateLookupTable":
                    state["input"] = None
            except Exception as error:
                signal_errors.append(error)
        connection.signal_subscribe(None, interface, None, path, None, Gio.DBusSignalFlags.NONE, received, None)
        call(path, interface, "SetCapabilities", GLib.Variant("(u)", (1 | 2 | 4 | 8,)))
        call(path, interface, "FocusIn")
        call("/org/freedesktop/IBus", "org.freedesktop.IBus", "SetGlobalEngine", GLib.Variant("(s)", ("yanliu",)))
        for child in pathlib.Path(f"/proc/{daemon.pid}/task/{daemon.pid}/children").read_text().split():
            command = pathlib.Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
            if command and command[0] == os.fsencode(prefix / "bin/yanliu-ibus"):
                service_pid = int(child)
        assert service_pid is not None
        # Two complete warm-up replays, then a fixed number of measured replays.
        for repeat in range(-2, args.repeats):
            for case in cases:
                state["target"] = None
                state["lookups"] = []
                call(path, interface, "Reset")
                wait(lambda: any(not visible for _preedit, _texts, visible in state["lookups"]))
                state["complete_ns"], state["target"] = None, case
                state["first_ns"] = {}
                last_begin = None
                for length, char in enumerate(case["input"], 1):
                    begin = time.perf_counter_ns()
                    handled = call(path, interface, "ProcessKeyEvent", GLib.Variant("(uuu)", (ord(char), 0, 0))).unpack()[0]
                    end = time.perf_counter_ns()
                    assert handled
                    drain()
                    prefix_input = case["input"][:length]
                    wait(lambda: prefix_input in state["first_ns"])
                    if repeat >= 0:
                        key_samples.append({"repeat": repeat, "input": case["input"], "prefix_length": length, "round_trip_us": (end-begin)/1000, "first_candidates_us": (state["first_ns"][prefix_input]-begin)/1000})
                    last_begin = begin
                    drain()
                wait(lambda: state["complete_ns"] is not None)
                assert any(preedit == case["input"] and texts == case["expected"] for preedit, texts, _visible in state["lookups"])
                if repeat >= 0 and case["dictionary"] != case["expected"]:
                    final_samples.append({"repeat": repeat, "input": case["input"], "from_last_key_begin_us": (state["complete_ns"]-last_begin)/1000,
                                          "dictionary_first": case["dictionary"][0], "model_first": case["expected"][0]})
        call(path, "org.freedesktop.IBus.Service", "Destroy")
    finally:
        if service_pid is not None:
            try:
                os.kill(service_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait()
        if service_pid is not None:
            wait(lambda: not pathlib.Path(f"/proc/{service_pid}").exists())
        log.seek(0)
        daemon_output = log.read()

report = {"measurement": "private real IBus InputContext synchronous ProcessKeyEvent round trip; eventual promoted candidate signal received via official GI",
          "build": "native release (verify build command separately)", "cpu": next((line.split(":", 1)[1].strip() for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor()), "machine": platform.machine(), "kernel": platform.release(),
          "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(), "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
          "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT / "src/ibus.rs", ROOT / "src/bin/yanliu-ibus.rs", ROOT / "src/model.rs"]}, "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
          "repeats": args.repeats, "warmup_replays": 2, "cases": cases, "key_round_trip": summary([sample["round_trip_us"] for sample in key_samples]),
          "first_dictionary_candidates": summary([sample["first_candidates_us"] for sample in key_samples]),
          "eventual_model_from_last_key_begin": summary([sample["from_last_key_begin_us"] for sample in final_samples]),
          "raw_key_samples": key_samples, "raw_model_samples": final_samples, "daemon_log": daemon_output}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(args.output), "key_round_trip": report["key_round_trip"], "first_dictionary_candidates": report["first_dictionary_candidates"], "eventual_model_from_last_key_begin": report["eventual_model_from_last_key_begin"]}, ensure_ascii=False))
