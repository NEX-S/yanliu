#!/usr/bin/python3
"""Check the IBus service through private ibus-daemons with temporary XDG state.

Never connects to the desktop IBus, selects input sources or reads user configuration.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import time
from xml.sax.saxutils import escape

import gi

gi.require_version('IBus', '1.0')
from gi.repository import Gio, GLib, IBus

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'data/models/sentence-model.safetensors'
IBUS, IBUS_PATH = 'org.freedesktop.IBus', '/org/freedesktop/IBus'
YANLIU, FACTORY = 'org.freedesktop.IBus.Yanliu', '/org/freedesktop/IBus/Factory'
ENGINE, CONTEXT = 'org.freedesktop.IBus.Engine', 'org.freedesktop.IBus.InputContext'
SERVICE, PROPERTIES = 'org.freedesktop.IBus.Service', 'org.freedesktop.DBus.Properties'
SHIFT, LOCK, CONTROL, ALT, NUM_LOCK = 1, 2, 4, 8, 16
MOD4, SUPER, HYPER, META, RELEASE = 1 << 6, 1 << 26, 1 << 27, 1 << 28, 1 << 30
PASSWORD_PURPOSE = 8


def wait(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('timed out waiting for private IBus state')


def settle(signals, quiet=0.1):
    """Wait until QUIET seconds pass without a new signal."""
    seen = {}

    def quiet_enough():
        if seen.get('count') != len(signals):
            seen.update(count=len(signals), since=time.monotonic())
        return time.monotonic() - seen['since'] >= quiet

    wait(quiet_enough)


def stop(process, group=False):
    def send(number):
        if group:
            try:
                os.killpg(process.pid, number)
            except ProcessLookupError:
                pass
        else:
            process.send_signal(number)

    send(signal.SIGTERM)
    try:
        process.wait(timeout=10)
        if group:
            wait(lambda: not process_group_alive(process.pid), timeout=10)
    except (subprocess.TimeoutExpired, AssertionError):
        send(signal.SIGKILL)
        process.wait()
        if group:
            wait(lambda: not process_group_alive(process.pid), timeout=10)
        raise


def process_group_alive(group):
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


class PrivateBus:
    """ibus-daemon on a private socket; services, logs and XDG state share one temporary directory."""

    def __init__(self, component_path=None):
        self.directory = Path(tempfile.mkdtemp(prefix='yanliu-ibus-'))
        self.address = f'unix:path={self.directory}/bus'
        self.env = dict(os.environ, IBUS_ADDRESS=self.address, **{
            f'XDG_{kind}_HOME': str(self.directory / kind.lower()) for kind in ('CONFIG', 'STATE', 'CACHE', 'DATA')})
        if component_path is not None:
            self.env['IBUS_COMPONENT_PATH'] = str(component_path)
        self.daemon = None
        self.services = []

    def __enter__(self):
        try:
            with (self.directory / 'daemon.log').open('w') as log:
                # Services the daemon auto-starts join this process group.
                self.daemon = subprocess.Popen(
                    ['ibus-daemon', '--single', '--panel=disable', '--config=disable', '--emoji-extension=disable',
                     '--cache=none', '--address=' + self.address],
                    env=self.env, stdout=log, stderr=log, process_group=0)
            wait(lambda: (self.directory / 'bus').exists())
            self.connection = Gio.DBusConnection.new_for_address_sync(
                self.address, Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT
                | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION, None, None)
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def __exit__(self, kind, error, traceback):
        failures = []
        try:
            if error is not None:
                for log in sorted(self.directory.glob('*.log')):
                    print(f'--- {log.name}\n{log.read_text()}', file=sys.stderr)
        except OSError as failure:
            failures.append(failure)
        for process in [*reversed(self.services), *([self.daemon] if self.daemon is not None else [])]:
            try:
                stop(process, group=process is self.daemon)
            except BaseException as failure:
                failures.append(failure)
        try:
            shutil.rmtree(self.directory)
        except OSError as failure:
            failures.append(failure)
        if error is not None:
            for failure in failures:
                error.add_note(f'private IBus cleanup: {failure!r}')
        elif failures:
            for failure in failures[1:]:
                failures[0].add_note(f'private IBus cleanup: {failure!r}')
            raise failures[0]

    def call(self, destination, path, interface, method, parameters=None):
        return self.connection.call_sync(destination, path, interface, method, parameters, None,
                                         Gio.DBusCallFlags.NONE, 15000, None)

    def name_has_owner(self, name):
        return self.call('org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus', 'NameHasOwner',
                         GLib.Variant('(s)', (name,))).unpack()[0]

    def start_service(self, binary, *options):
        wait(lambda: not self.name_has_owner(YANLIU))
        log = self.directory / f'service-{len(self.services) + 1}.log'
        with log.open('w') as stream:
            service = subprocess.Popen([binary, '--ibus-address', self.address, *options],
                                       env=self.env, stdout=stream, stderr=stream)
        service.log = log
        self.services.append(service)
        wait(lambda: service.poll() is not None or 'factory ready' in log.read_text())
        assert service.poll() is None, log.read_text()
        return service

    def startup_error(self, binary, *options):
        result = subprocess.run([binary, '--ibus-address', self.address, *options],
                                env=self.env, capture_output=True, text=True, timeout=20)
        assert result.returncode != 0, result.stderr
        return result.stderr

    def record(self, interface, path):
        signals = []
        self.connection.signal_subscribe(None, interface, None, path, None, Gio.DBusSignalFlags.NONE,
                                         lambda _c, _s, _p, _i, name, params, _d: signals.append((name, params)), None)
        return signals

    def engine(self, path, method, parameters=None, interface=ENGINE):
        return self.call(YANLIU, path, interface, method, parameters)

    def create_engine(self):
        return self.engine(FACTORY, 'CreateEngine', GLib.Variant('(s)', ('yanliu',)),
                           'org.freedesktop.IBus.Factory').unpack()[0]

    def engine_key(self, path, value, modifiers=0):
        return self.engine(path, 'ProcessKeyEvent', GLib.Variant('(uuu)', (value, 0, modifiers))).unpack()[0]

    def input_context(self):
        path = self.call(IBUS, IBUS_PATH, IBUS, 'CreateInputContext', GLib.Variant('(s)', ('yanliu-test',))).unpack()[0]
        signals = self.record(CONTEXT, path)
        # PREEDIT_TEXT | AUXILIARY_TEXT | LOOKUP_TABLE | FOCUS: receive native candidate messages.
        self.call(IBUS, path, CONTEXT, 'SetCapabilities', GLib.Variant('(u)', (1 | 2 | 4 | 8,)))
        self.call(IBUS, path, CONTEXT, 'FocusIn')
        self.call(IBUS, IBUS_PATH, IBUS, 'SetGlobalEngine', GLib.Variant('(s)', ('yanliu',)))
        engine = self.call(IBUS, path, CONTEXT, 'GetEngine').get_child_value(0).get_variant()
        assert engine.get_child_value(2).get_string() == 'yanliu'
        return path, signals

    def context_key(self, path, value, modifiers=0):
        return self.call(IBUS, path, CONTEXT, 'ProcessKeyEvent', GLib.Variant('(uuu)', (value, 0, modifiers))).unpack()[0]


def ibus_text(value):
    return GLib.Variant('(sa{sv}sv)', ('IBusText', {}, value, GLib.Variant('(sa{sv}av)', ('IBusAttrList', {}, []))))


def ibus_object(params):
    return IBus.Serializable.deserialize_object(params.get_child_value(0).get_variant())


def last_table(signals):
    return ibus_object(next(p for name, p in reversed(signals) if name == 'UpdateLookupTable'))


def candidates(table):
    return [table.get_candidate(i).get_text() for i in range(table.get_number_of_candidates())]


def commits(signals):
    return [ibus_object(p).get_text() for name, p in signals if name == 'CommitText']


def visible_table(signals):
    return any(name == 'UpdateLookupTable' and p.get_child_value(1).get_boolean() for name, p in signals)


def learned(path):
    return {(row['input'], row['text']): row['count'] for row in json.loads(path.read_text())['rows']}


def panel_state(params):
    """Raw pinyin and caret shown in the auxiliary text; explanations follow '；'."""
    text = ibus_object(params).get_text()
    if not params.get_child_value(1).get_boolean():
        assert text == '', text
        return '', 0, False
    assert text.count('▏') == 1, text
    before, after = text.partition('；')[0].split('▏')
    raw = before + after
    assert raw and all(c in string.ascii_letters + "'" for c in raw), text
    return raw, len(before), True


def shows_panel(signals, state):
    return any(name == 'UpdateAuxiliaryText' and panel_state(p) == state for name, p in signals)


def assert_inline_hidden(signals):
    """Applications only receive an empty hidden preedit; the panel shows the composition."""
    for name, params in signals:
        if name == 'UpdatePreeditText':
            assert ibus_object(params).get_text() == ''
            assert params.get_child_value(1).get_uint32() == 0
            assert not params.get_child_value(2).get_boolean()
            # The Engine signal carries focus mode; InputContext forwards only (vub).
            if params.n_children() == 4:
                assert params.get_child_value(3).get_uint32() == 0


def check_protocol(binary, options):
    with PrivateBus() as bus:
        service = bus.start_service(binary, '--register', '--data-dir', ROOT / 'data', *options)
        first, second = bus.create_engine(), bus.create_engine()
        assert first != second
        assert bus.engine_key(first, ord('n'))
        assert not bus.engine_key(second, ord(' ')), 'engine sessions must be isolated'
        bus.engine(first, 'FocusOutId', GLib.Variant('(s)', ('/test/focus',)))
        assert not bus.engine_key(first, ord(' ')), 'focus out must clear the composition'
        bus.engine(first, 'SetSurroundingText', GLib.Variant('(vuu)', (ibus_text('你好'), 2, 2)))
        try:
            bus.engine(first, 'SetSurroundingText', GLib.Variant('(vuu)', (ibus_text('你好'), 3, 3)))
            raise AssertionError('a cursor beyond the Unicode text must be rejected')
        except GLib.Error as error:
            assert 'InvalidArgs' in str(error), error

        first_signals = bus.record(ENGINE, first)
        for char in 'ni':
            bus.engine_key(first, ord(char))
        wait(lambda: shows_panel(first_signals, ('ni', 2, True)))

        def wait_cursor(start, position):
            wait(lambda: any(name == 'UpdateLookupTable' and ibus_object(p).get_cursor_pos() == position
                             for name, p in first_signals[start:]))

        start = len(first_signals)
        bus.engine(first, 'CursorUp')
        wait_cursor(start, 0)
        table = last_table(first_signals)
        assert table.get_number_of_candidates() > 9, 'paging needs a second page'
        second_page_first = table.get_candidate(9).get_text()
        for value, modifiers, position in [(ord('j'), CONTROL, 9), (ord('k'), CONTROL, 0),
                                           (ord('h'), CONTROL, table.get_number_of_candidates() - 1),
                                           (ord('l'), CONTROL, 0), (IBus.KEY_Page_Down, 0, 9), (IBus.KEY_Page_Up, 0, 0)]:
            start = len(first_signals)
            assert bus.engine_key(first, value, modifiers)
            wait_cursor(start, position)
        start = len(first_signals)
        bus.engine(first, 'PageDown')
        wait_cursor(start, 9)
        bus.engine(first, 'CandidateClicked', GLib.Variant('(uuu)', (0, 1, 0)))
        wait(lambda: commits(first_signals))
        assert commits(first_signals) == [second_page_first]

        def assert_preference(engine):
            signals = bus.record(ENGINE, engine)
            for char in 'ni':
                bus.engine_key(engine, ord(char))
            # Navigation cancels asynchronous ranking before the learned order is read.
            bus.engine(engine, 'CursorDown')
            wait(lambda: visible_table(signals))
            assert last_table(signals).get_candidate(0).get_text() == second_page_first
            bus.engine(engine, 'Reset')

        assert_preference(second)
        bus.engine(first, 'Destroy', interface=SERVICE)
        replacement = bus.create_engine()
        assert_preference(replacement)
        for char in 'ni':  # TERM arrives while this engine still composes.
            bus.engine_key(replacement, ord(char))
        bus.engine(second, 'Destroy', interface=SERVICE)
        service.terminate()
        assert service.wait(timeout=5) == 0, 'TERM must finish learning writes'
        learning_file = bus.directory / 'state/yanliu/learning.json'
        assert learned(learning_file)[('ni', second_page_first)] == 1
        service = bus.start_service(binary, '--register', '--data-dir', ROOT / 'data', *options)
        restored = bus.create_engine()
        assert_preference(restored)
        bus.engine(restored, 'Destroy', interface=SERVICE)

        path, signals = bus.input_context()

        def key(value, modifiers=0):
            return bus.context_key(path, value, modifiers)

        def set_purpose(purpose):
            bus.call(IBUS, path, PROPERTIES, 'Set',
                     GLib.Variant('(ssv)', (CONTEXT, 'ContentType', GLib.Variant('(uu)', (purpose, 0)))))

        assert not key(ord('n'), RELEASE), 'release must pass through'
        assert not key(ord('n'), CONTROL), 'Ctrl shortcuts must pass through'
        for char in 'nihao':
            assert key(ord(char))
        wait(lambda: visible_table(signals))
        table = last_table(signals)
        assert table.get_orientation() == IBus.Orientation.HORIZONTAL
        assert table.get_candidate(0).get_text() == '你好', table.get_candidate(0).get_text()
        panel = next(p for name, p in reversed(signals) if name == 'UpdateAuxiliaryText')
        assert panel_state(panel) == ('nihao', 5, True)
        assert key(ord(' '))
        wait(lambda: commits(signals))
        assert commits(signals) == ['你好']
        assert key(ord('n'))
        start = len(signals)
        bus.call(IBUS, path, CONTEXT, 'Reset')
        wait(lambda: shows_panel(signals[start:], ('', 0, False)))
        assert not key(ord(' ')), 'reset must clear the composition'
        for char in 'hjkl,.':
            assert not key(ord(char), CONTROL), 'idle Ctrl shortcuts must pass through'

        def compose(text):
            start = len(signals)
            for char in text:
                assert key(ord(char))
            wait(lambda: shows_panel(signals[start:], (text, len(text), True)))
            # Navigation cancels pending ranking and publishes a stable candidate list.
            assert key(IBus.KEY_Up)

            def table():
                raw = found = None
                for name, params in signals[start:]:
                    if name == 'UpdateAuxiliaryText':
                        raw = panel_state(params)
                    elif name == 'UpdateLookupTable' and raw == (text, len(text), True):
                        found = ibus_object(params)
                return found

            wait(lambda: table() is not None)
            return table()

        def select_partial(text, prefix):
            menu = candidates(compose(text))
            assert prefix in menu, f'{text} needs the partial candidate {prefix}'
            for _ in range(menu.index(prefix)):
                assert key(IBus.KEY_Down)

        def edit_key(value, raw, cursor, expected_commits=()):
            start = len(signals)
            assert key(value)
            wait(lambda: shows_panel(signals[start:], (raw, cursor, bool(raw))))
            assert commits(signals[start:]) == list(expected_commits), (value, commits(signals[start:]))

        def punctuation(char, expected, modifiers=0):
            start = len(signals)
            assert key(ord(char), modifiers)
            wait(lambda: shows_panel(signals[start:], ('', 0, False)))
            assert commits(signals[start:]) == expected, (char, commits(signals[start:]), expected)
            assert not key(ord(' ')), 'punctuation must leave no composition'

        assert compose("NI'HAO").get_candidate(0).get_text() == '你好'
        edit_key(IBus.KEY_Return, '', 0, ["NI'HAO"])
        compose('nihaoq')
        for value, raw, cursor in [
                (IBus.KEY_Left, 'nihaoq', 5), (IBus.KEY_BackSpace, 'nihaq', 4), (ord('o'), 'nihaoq', 5),
                (IBus.KEY_Delete, 'nihao', 5), (IBus.KEY_Right, 'nihao', 5), (IBus.KEY_Home, 'nihao', 0),
                (IBus.KEY_Delete, 'ihao', 0), (ord('n'), 'nihao', 1), (IBus.KEY_End, 'nihao', 5),
                (IBus.KEY_Left, 'nihao', 4), (IBus.KEY_BackSpace, 'niho', 3), (ord('a'), 'nihao', 4),
                (IBus.KEY_Right, 'nihao', 5)]:
            edit_key(value, raw, cursor)
        edit_key(IBus.KEY_Return, '', 0, ['nihao'])
        compose('nihao')
        edit_key(IBus.KEY_Escape, '', 0)
        select_partial("ni'haoq", '你好')
        edit_key(ord(' '), 'q', 1, ['你好'])
        edit_key(IBus.KEY_Return, '', 0, ['q'])

        special = dict(zip(',.!?:;[]<>\\()', '，。！？：；【】《》、（）'))
        for char in string.punctuation:
            punctuation(char, [special.get(char, char)])
        compose('haoq')
        for char in 'hjkl':
            for extra in [SHIFT, ALT, MOD4, SUPER, RELEASE]:
                assert not key(ord(char), CONTROL | extra), 'modified Ctrl navigation must pass through'
        for char in ',.':
            assert not key(ord(char), CONTROL), 'Ctrl punctuation belongs to the application'
        for extra in [SHIFT, ALT, MOD4, SUPER, HYPER, META, RELEASE]:
            assert not key(ord(' '), CONTROL | extra), 'modified or released Ctrl+Space must pass through'
        start = len(signals)
        assert key(ord(' '), CONTROL | LOCK | NUM_LOCK), 'Ctrl+Space ignores lock modifiers'
        wait(lambda: shows_panel(signals[start:], ('', 0, False)))
        assert commits(signals[start:]) == ['haoq'], 'switching to English commits the raw input unselected'
        for char in 'nihao' + string.punctuation + ' ':
            assert not key(ord(char)), 'English mode passes typing through'
        for char in 'hjkl,.':
            assert not key(ord(char), CONTROL), 'English Ctrl shortcuts pass through'
        bus.call(IBUS, path, CONTEXT, 'Reset')
        bus.call(IBUS, path, CONTEXT, 'FocusOut')
        bus.call(IBUS, path, CONTEXT, 'FocusIn')
        assert not key(ord('n')), 'focus and reset keep English mode'
        assert key(ord(' '), CONTROL), 'Ctrl+Space returns to Chinese'
        punctuation('-', ['-'])
        punctuation('=', ['='])

        for ascii_char, chinese_char in zip(',.!?:;', '，。！？：；'):
            assert compose('nihao').get_candidate(0).get_text() == '你好'
            for modifier in [RELEASE, CONTROL, ALT, MOD4, SUPER, HYPER, META]:
                start = len(signals)
                assert not key(ord(ascii_char), modifier)
                assert not commits(signals[start:])
            punctuation(ascii_char, ['你好', chinese_char], SHIFT if ascii_char in '!?:' else 0)
        table = compose('nihao')
        assert key(IBus.KEY_Down)
        punctuation(',', [table.get_candidate(1).get_text(), '，'])
        for char, expected in [('.', '。'), ('\\', '、'), ('/', '/')]:
            select_partial('nihaoq', '你好')
            punctuation(char, ['你好', 'q', expected])
        assert compose("nihao''").get_number_of_candidates() == 0, 'invalid syllable boundaries leave raw input'
        for char in 'hjkl':
            assert not key(ord(char), CONTROL), 'Ctrl navigation without candidates passes through'
        punctuation('?', ["nihao''", '？'])

        start = len(signals)
        for char in 'ni':
            assert key(ord(char))
        wait(lambda: shows_panel(signals[start:], ('ni', 2, True)))
        start = len(signals)
        set_purpose(PASSWORD_PURPOSE)
        wait(lambda: shows_panel(signals[start:], ('', 0, False)) and any(
            name == 'UpdateLookupTable' and not p.get_child_value(1).get_boolean() for name, p in signals[start:]))
        assert not commits(signals[start:]), 'a password field hides the composition without committing'
        assert not key(ord('n')), 'password fields pass typing through'
        assert not key(ord(' '), CONTROL), 'password fields keep the input mode'
        for char in 'hjkl':
            assert not key(ord(char), CONTROL), 'password navigation passes through'
        for char in ',.!?:;-=':
            assert not key(ord(char)), 'password punctuation passes through'
        set_purpose(0)
        punctuation(',', ['，'])
        bus.call(IBUS, path, CONTEXT, 'FocusOut')
        bus.call(IBUS, path, SERVICE, 'Destroy')
        service.terminate()
        assert service.wait(timeout=5) == 0
        rows = learned(learning_file)
        assert rows[('ni', second_page_first)] == 1, 'retyping, resets and password mode must not learn'
        assert rows[('nihao', '你好')] == 10, 'space, six punctuation and three partial choices learn once each'
        assert rows[("ni'hao", '你好')] == 1, 'a partial choice learns exactly the consumed raw bytes'
        assert not {'haoq', 'q', 'nihaoq', "nihao''", "ni'haoq"} & {raw for raw, _ in rows}, \
            'raw commits and unconsumed remainders must not learn'
        assert_inline_hidden(first_signals + signals)


def component_fixture(directory, binary):
    """The production XML template around BINARY, with a config path the XML must quote."""
    prefix = directory / 'prefix'
    (prefix / 'bin').mkdir(parents=True)
    (prefix / 'bin/yanliu-ibus').symlink_to(binary)
    config = directory / 'config with spaces/砚流/config.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'data_dir': str(ROOT / 'data')}))
    component_path = prefix / 'share/ibus/component'
    component_path.mkdir(parents=True)
    (component_path / 'yanliu.xml').write_text((ROOT / 'packaging/yanliu.xml.in').read_text()
                                                .replace('@PREFIX@', str(prefix))
                                                .replace('@CONFIG@', escape(shlex.quote(str(config)))))
    return component_path


def check_component(component_path):
    with PrivateBus(component_path) as bus:
        engines = bus.call(IBUS, IBUS_PATH, IBUS, 'ListEngines').get_child_value(0)
        names = [IBus.Serializable.deserialize_object(engines.get_child_value(i).get_variant()).get_name()
                 for i in range(engines.n_children())]
        assert 'yanliu' in names, names
        assert not bus.name_has_owner(YANLIU), 'the daemon must start the service on demand'
        path, signals = bus.input_context()
        for char in 'nihao':
            assert bus.context_key(path, ord(char))
        wait(lambda: shows_panel(signals, ('nihao', 5, True)))
        assert last_table(signals).get_candidate(0).get_text() == '你好'
        assert bus.context_key(path, ord(' '))
        wait(lambda: commits(signals))
        assert commits(signals) == ['你好']
        assert any(name == 'UpdatePreeditText' for name, _ in signals)
        assert_inline_hidden(signals)
        bus.call(IBUS, path, SERVICE, 'Destroy')


def check_learning_failures(binary):
    with PrivateBus() as bus:
        state = bus.directory / 'state/yanliu'
        state.mkdir(parents=True)
        learning_file, lock_file = state / 'learning.json', state / 'learning.lock'
        options = ('--data-dir', ROOT / 'data')
        learning_file.write_text('corrupt')
        assert 'expected value' in bus.startup_error(binary, *options)
        learning_file.unlink()
        os.mkfifo(learning_file)
        assert 'must be a regular file' in bus.startup_error(binary, *options), 'a FIFO must fail without blocking'
        learning_file.unlink()
        with lock_file.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            error = bus.startup_error(binary, *options)
            assert 'lock learning state' in error and ('block' in error.lower() or 'locked' in error.lower()), error

        service = bus.start_service(binary, *options)
        engine = bus.create_engine()
        learning_file.mkdir()
        for value in (ord('n'), ord('i'), ord(' ')):
            assert bus.engine_key(engine, value)
        assert service.wait(timeout=20) != 0
        cause = service.log.read_text()
        assert 'learning writer' in cause and 'Is a directory' in cause, cause
        assert '"ni"' not in cause and '你' not in cause, 'errors must not log input or text'

        # Disabled learning bypasses parsing and the lock while still converting real keys.
        learning_file.rmdir()
        learning_file.write_bytes(b'corrupt disabled-learning snapshot')
        lock_file.write_bytes(b'parent owns disabled-learning lock')

        def persisted():
            return [(path.name, path.read_bytes(), path.stat().st_ino, path.stat().st_mode, path.stat().st_mtime_ns)
                    for path in sorted(state.iterdir())]

        before = persisted()
        with lock_file.open('r+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            service = bus.start_service(binary, *options, '--no-learning')
            engine = bus.create_engine()
            signals = bus.record(ENGINE, engine)
            bus.engine(engine, 'FocusIn')
            for value in (ord('n'), ord('i'), ord(' ')):
                assert bus.engine_key(engine, value)
            wait(lambda: commits(signals))
            assert commits(signals)[0] not in ('', 'ni'), 'disabled learning still commits a conversion'
            assert persisted() == before, 'disabled learning changed persistent files or the lock'
            service.terminate()
            assert service.wait(timeout=20) == 0, service.log.read_text()
            assert persisted() == before, 'disabled learning changed state during shutdown'


def check_user_dictionary(binary):
    with PrivateBus() as bus:
        dictionaries = bus.directory / 'assets/dictionaries'
        dictionaries.mkdir(parents=True)
        for name in ('8105.dict.yaml', 'base.dict.yaml', 'ext.dict.yaml'):
            (dictionaries / name).write_text('...\n你\tni\t1\n')
        words = bus.directory / 'words.dict.yaml'
        words.write_text('...\n砚流星舟\tyan liu xing zhou\t100\n')
        config = bus.directory / 'config.json'
        config.write_text(json.dumps({'data_dir': 'assets', 'model': str(MODEL),
                                      'user_dictionary': 'words.dict.yaml', 'learning_enabled': False}))
        bus.start_service(binary, '--config', config)
        engine = bus.create_engine()
        signals = bus.record(ENGINE, engine)
        for char in 'yanliuxingzhou ':
            assert bus.engine_key(engine, ord(char))
        wait(lambda: commits(signals))
        assert commits(signals) == ['砚流星舟'], commits(signals)
        words.write_text('...\n坏词\tBAD\n')
        error = bus.startup_error(binary, '--config', config)
        assert str(words) in error and 'line: 2' in error and 'invalid pinyin' in error, error
        assert not (bus.directory / 'state').exists(), 'disabled learning created state'


def check_contextual_learning(binary, model):
    with PrivateBus() as bus:
        dictionaries = bus.directory / 'assets/dictionaries'
        dictionaries.mkdir(parents=True)
        (dictionaries / '8105.dict.yaml').write_text('...\n现\txian\t1000\n西安\txi an\t100\n')
        for name in ('base.dict.yaml', 'ext.dict.yaml'):
            (dictionaries / name).write_text('...\n')
        config = bus.directory / 'config.json'
        settings = {'data_dir': str(dictionaries.parent), 'model': str(model), 'contextual_learning': True}
        config.write_text(json.dumps(settings))
        snapshot_file = bus.directory / 'state/yanliu/learning.json'
        choices = [('乘火车前往', '西安'), ('此时此刻', '现')]

        def launch():
            service = bus.start_service(binary, '--config', config)
            engine = bus.create_engine()
            return service, engine, bus.record(ENGINE, engine)

        def stop_service(service):
            service.terminate()
            assert service.wait(timeout=20) == 0, service.log.read_text()

        def menu(signals):
            return candidates(last_table(signals)) if visible_table(signals) else []

        def type_xian(engine, signals, context):
            """Type after CONTEXT and return the index of this input's first signal."""
            bus.engine(engine, 'FocusIn')
            bus.engine(engine, 'SetSurroundingText', GLib.Variant('(vuu)', (ibus_text(context), len(context), len(context))))
            start = len(signals)
            for char in 'xian':
                assert bus.engine_key(engine, ord(char))
            wait(lambda: {'现', '西安'} <= set(menu(signals[start:])))
            # Let queued model reorderings arrive; this synchronizes, it does not measure latency.
            settle(signals)
            return start

        def assert_contextual_first_choices(engine, signals):
            for context, expected in choices:
                start = type_xian(engine, signals, context)
                wait(lambda: menu(signals[start:])[0] == expected)

        service, engine, signals = launch()
        for _ in range(4):
            for context, expected in choices:
                type_xian(engine, signals, context)
                start = len(signals)
                bus.engine(engine, 'CursorDown')  # Navigation freezes the visible order before the click.
                wait(lambda: any(name == 'UpdateLookupTable' for name, _ in signals[start:]))
                selected = menu(signals).index(expected)
                start = len(signals)
                bus.engine(engine, 'CandidateClicked', GLib.Variant('(uuu)', (selected, 1, 0)))
                wait(lambda: commits(signals[start:]))
                assert commits(signals[start:]) == [expected]
        assert_contextual_first_choices(engine, signals)
        stop_service(service)
        snapshot = json.loads(snapshot_file.read_text())
        assert sorted((row['text'], row['count']) for row in snapshot['rows']) == [('现', 4), ('西安', 4)]
        assert all(row['input'] == 'xian' for row in snapshot['rows'])
        assert snapshot['updates'] > 0
        assert not any(context in snapshot_file.read_text() for context, _ in choices), 'contexts must not be stored'

        service, engine, signals = launch()
        assert_contextual_first_choices(engine, signals)
        bus.engine(engine, 'Set', GLib.Variant('(ssv)', (ENGINE, 'ContentType', GLib.Variant('(uu)', (PASSWORD_PURPOSE, 0)))),
                   PROPERTIES)
        private_context = '敏感前文不得保存'
        bus.engine(engine, 'SetSurroundingText',
                   GLib.Variant('(vuu)', (ibus_text(private_context), len(private_context), len(private_context))))
        for char in 'xian ':
            assert not bus.engine_key(engine, ord(char)), 'sensitive fields pass typing through'
        stop_service(service)
        assert json.loads(snapshot_file.read_text()) == snapshot, 'sensitive input must not learn'

        settings['contextual_learning'] = False
        config.write_text(json.dumps(settings))
        service, engine, signals = launch()
        for context, _ in choices:
            start = type_xian(engine, signals, context)
            assert menu(signals[start:])[0] == '现', menu(signals[start:])
        stop_service(service)
        assert json.loads(snapshot_file.read_text()) == snapshot, 'disabled contextual learning must not update'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path, help='yanliu-ibus executable under test')
    parser.add_argument('--model', help='service model for the protocol and contextual learning checks')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'npu'], help='service --device for the protocol check')
    parser.add_argument('--runtime-dir', help='service --runtime-dir for the protocol check')
    parser.add_argument('--installed', type=Path, metavar='PREFIX',
                        help='check the component installed under PREFIX instead of a fixture around BINARY')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # Clean up private processes on TERM as on Ctrl+C.
    binary = args.binary.resolve(strict=True)
    options = []
    for flag, value in (('--model', args.model), ('--device', args.device), ('--runtime-dir', args.runtime_dir)):
        if value is not None:
            options += [flag, value]
    check_protocol(binary, options)
    print('PASS: engines, focus, surrounding text, paging and clicks, learning across engines and restart, '
          'editing, partial selection, punctuation, English mode, reset, password passthrough')
    if args.installed is None:
        with tempfile.TemporaryDirectory(prefix='yanliu-component-') as directory:
            check_component(component_fixture(Path(directory), binary))
    else:
        check_component(args.installed.resolve(strict=True) / 'share/ibus/component')
    print('PASS: component discovery and on-demand service start')
    check_learning_failures(binary)
    print('PASS: learning startup and writer failures keep their causes; --no-learning leaves state untouched')
    check_user_dictionary(binary)
    print('PASS: explicit user dictionary; malformed table rejected with its path')
    check_contextual_learning(binary, Path(args.model).resolve(strict=True) if args.model else MODEL)
    print('PASS: contextual corrections persist across restart; sensitive input and opt-out do not learn')


if __name__ == '__main__':
    main()
