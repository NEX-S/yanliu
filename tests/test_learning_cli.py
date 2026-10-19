"""Exercise learning management through the built CLI, without dictionary/model assets."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BINARY = Path(os.environ.get(
    'YANLIU_BIN', Path(__file__).resolve().parents[1] / 'target/debug/yanliu'
)).resolve()


class LearningCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='yanliu-learning-cli-')
        self.addCleanup(self.temporary.cleanup)
        self.state_home = Path(self.temporary.name)
        self.directory = self.state_home / 'yanliu'
        self.path = self.directory / 'learning.json'

    def run_cli(self, command):
        return subprocess.run(
            [str(BINARY), '--dictionary-dir', str(self.state_home / 'missing-dictionary'),
             '--model', str(self.state_home / 'missing-model'), 'learning', command],
            env={**os.environ, 'XDG_STATE_HOME': str(self.state_home)},
            text=True, capture_output=True, timeout=10,
        )

    def success(self, command):
        result = self.run_cli(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value['path'], str(self.path))
        return value['rows']

    def test_missing_state_list_and_idempotent_clear_without_assets(self):
        self.assertEqual(self.success('list'), [])
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.success('clear'), [])
        lock_inode = (self.directory / 'learning.lock').stat().st_ino
        self.assertEqual(self.success('clear'), [])
        self.assertEqual((self.directory / 'learning.lock').stat().st_ino, lock_inode)
        self.assertEqual(json.loads(self.path.read_text()),
                         {'version': 1, 'rows': [], 'adapter': None, 'updates': 0})
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_list_persisted_rows_validation_and_corruption_recovery(self):
        self.directory.mkdir()
        rows = [{'input': 'GPU', 'text': 'GPU', 'count': 1},
                {'input': 'ni', 'text': '泥', 'count': 3}]
        self.path.write_text(json.dumps(rows))
        self.assertEqual(self.success('list'), rows)
        for body in ['broken', '[{"input":"n1","text":"泥","count":1}]',
                     '[{"input":"ni","text":"泥","count":0}]',
                     json.dumps(rows + rows)]:
            with self.subTest(body=body):
                self.path.write_text(body)
                result = self.run_cli('list')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(self.path), result.stderr)
                self.assertEqual(self.success('clear'), [])
                self.assertEqual(json.loads(self.path.read_text()),
                                 {'version': 1, 'rows': [], 'adapter': None, 'updates': 0})

    def test_clear_refuses_active_owner_and_preserves_snapshot(self):
        self.directory.mkdir()
        original = b'[{"input":"ni","text":"you","count":1}]'
        self.path.write_bytes(original)
        lock_path = self.directory / 'learning.lock'
        with lock_path.open('a+b') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_cli('clear')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('stop yanliu-ibus and retry', result.stderr)
            self.assertEqual(self.path.read_bytes(), original)
            self.assertEqual(lock_path.stat().st_ino, os.fstat(owner.fileno()).st_ino)
        self.assertEqual(self.success('clear'), [])


if __name__ == '__main__':
    unittest.main()
