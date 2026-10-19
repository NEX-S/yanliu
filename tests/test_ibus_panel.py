#!/usr/bin/python3
"""Wire contract for raw composition in Yanliu's IBus panel."""
import unittest
import gi

gi.require_version('IBus', '1.0')
from gi.repository import GLib
from ibus_panel import panel_state, assert_inline_hidden


def text(value):
    attributes = GLib.Variant('(sa{sv}av)', ('IBusAttrList', {}, []))
    return GLib.Variant('(sa{sv}sv)', ('IBusText', {}, value, attributes))


def auxiliary(value, visible=True):
    return GLib.Variant('(vb)', (text(value), visible))


class PanelWireTest(unittest.TestCase):
    def test_raw_cursor_and_explanations(self):
        for shown, expected in [
            ('nihao▏', ('nihao', 5, True)),
            ('ni▏hao', ('nihao', 2, True)),
            ('▏nihao', ('nihao', 0, True)),
            ("ni'hao▏；部分转换：保留剩余拼音", ("ni'hao", 6, True)),
            ('nih▏；补全：包含未输入内容', ('nih', 3, True)),
            ('in▏；纠错：相邻字母换位', ('in', 2, True)),
            ('GPU▏jiasu', ('GPUjiasu', 3, True)),
        ]:
            with self.subTest(shown=shown):
                self.assertEqual(panel_state(auxiliary(shown)), expected)
        self.assertEqual(panel_state(auxiliary('', False)), ('', 0, False))

    def test_invalid_panel_is_rejected(self):
        for shown, visible in [
            ('nihao', True), ('ni▏hao▏', True), ('n1▏', True),
            ('▏', True), ('ni▏', False),
        ]:
            with self.subTest(shown=shown, visible=visible):
                with self.assertRaises((AssertionError, ValueError)):
                    panel_state(auxiliary(shown, visible))

    def test_hidden_context_and_engine_preedit(self):
        for signature, values in [
            ('(vub)', (text(''), 0, False)),
            ('(vubu)', (text(''), 0, False, 0)),
        ]:
            with self.subTest(signature=signature):
                assert_inline_hidden(GLib.Variant(signature, values))

    def test_invalid_inline_is_rejected(self):
        for values in [
            (text('ni'), 0, False, 0), (text(''), 1, False, 0),
            (text(''), 0, True, 0), (text(''), 0, False, 1),
        ]:
            with self.subTest(values=values):
                with self.assertRaises(AssertionError):
                    assert_inline_hidden(GLib.Variant('(vubu)', values))


if __name__ == '__main__':
    unittest.main()
