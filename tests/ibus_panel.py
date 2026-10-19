"""Decode Yanliu's panel composition and verify hidden inline IBus wire text."""
import gi

gi.require_version("IBus", "1.0")
from gi.repository import IBus

def panel_state(params):
    text = IBus.Serializable.deserialize_object(params.get_child_value(0).get_variant()).get_text()
    visible = params.get_child_value(1).get_boolean()
    if not visible:
        assert text == "", (text, visible)
        return "", 0, False
    raw_caret = text.partition("；")[0]
    assert text.count("▏") == 1, text
    before, after = raw_caret.split("▏")
    raw = before + after
    assert raw and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'" for c in raw), text
    return raw, len(before), True

def assert_inline_hidden(params):
    assert IBus.Serializable.deserialize_object(params.get_child_value(0).get_variant()).get_text() == ""
    assert params.get_child_value(1).get_uint32() == 0
    assert not params.get_child_value(2).get_boolean()
    # The Engine signal carries focus mode; InputContext forwards only (vub).
    if params.n_children() == 4:
        assert params.get_child_value(3).get_uint32() == 0
