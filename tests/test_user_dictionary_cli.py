"""Explicit user dictionary boundary through the real CLI, with small main dictionaries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BINARY = Path(os.environ.get(
    'YANLIU_BIN', Path(__file__).resolve().parents[1] / 'target/debug/yanliu'
)).resolve()


class UserDictionaryCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='yanliu-user-dictionary-cli-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for name in ('8105.dict.yaml', 'base.dict.yaml', 'ext.dict.yaml'):
            (self.directory / name).write_text('...\n你\tni\t1\n', encoding='utf-8')
        self.user = self.directory / 'user.dict.yaml'

    def run_cli(self, raw='yanliu', enabled=True):
        args = [str(BINARY), '--dictionary-dir', str(self.directory)]
        if enabled:
            args += ['--user-dictionary', str(self.user)]
        return subprocess.run(args + ['decode', raw], capture_output=True, text=True, timeout=10)

    def test_enable_disable_duplicate_and_candidate_identity(self):
        self.user.write_text('---\nname: mine\n...\n砚流\tyan liu\t100\n砚流\tyan liu\t1\n', encoding='utf-8')
        disabled = self.run_cli(enabled=False)
        self.assertEqual(disabled.returncode, 0, disabled.stderr)
        self.assertNotIn('砚流', disabled.stdout)
        for raw, expected in [('yanliu', 'consumed=6\tExactDictionary'),
                              ("yan'liu'ni", 'consumed=8\tExactDictionary'),
                              ('yanli', 'consumed=5\tCompletion')]:
            with self.subTest(raw=raw):
                result = self.run_cli(raw)
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [r.split('\t')[1:] for r in result.stdout.splitlines() if '\t砚流\t' in r]
                # Text can occur at different raw consumption lengths. Duplicate
                # dictionary rows must not duplicate the actual candidate identity.
                identities = [(row[0], row[1], row[2]) for row in rows]
                self.assertEqual(len(identities), len(set(identities)), result.stdout)
                matching = [row for row in rows if '\t'.join(row[1:3]) == expected]
                self.assertEqual(len(matching), 1, result.stdout)
        self.user.unlink()
        self.assertEqual(self.run_cli(enabled=False).returncode, 0)
        failed = self.run_cli()
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn(str(self.user), failed.stderr)

    def test_empty_table_and_valid_weight_and_pinyin_boundaries(self):
        for body in ['...\n', '...\n# comment\n\n',
                     '...\n砚流\tyan liu\n', '...\n砚流\tyan liu\t\n',
                     '...\n砚流\tyan liu\t0%\n',
                     '...\n词\t' + 'a' * 128 + '\t1\n',
                     '...\n词\t' + 'a ' * 128 + '\t1\n',
                     '...\n' + '词\tci\n' * 10000,
                     '...\n#' + 'x' * (8 * 1024 * 1024 - 5)]:
            self.user.write_text(body, encoding='utf-8')
            result = self.run_cli('ni')
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_non_regular_file_is_rejected_without_blocking(self):
        self.user.mkdir()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(self.user), result.stderr)
        self.assertIn('regular file', result.stderr)
        self.user.rmdir()
        os.mkfifo(self.user)
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(self.user), result.stderr)
        self.assertIn('regular file', result.stderr)

    def test_bad_assets_retain_path_line_and_reason(self):
        rows = [('词\tCI', 'invalid pinyin'), ('词', 'TSV columns'),
                ('词\tci\t1\textra', 'TSV columns'),
                ('词\x00\tci', 'control character'), ('\tci', 'empty text'),
                ('词\t', 'empty text or pinyin'),
                ('词\tci\tNaN', 'finite and nonnegative'),
                ('词\tci\tinf', 'finite and nonnegative'),
                ('词\tci\t-1', 'finite and nonnegative'),
                ('词\tci\tbad', 'invalid weight'),
                ('词\t' + 'a' * 129, 'exceeds 128 bytes')]
        for row, reason in rows:
            with self.subTest(reason=reason, row=row):
                self.user.write_text('---\n...\n' + row + '\n', encoding='utf-8')
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(self.user), result.stderr)
                self.assertIn('line: 3', result.stderr)
                self.assertIn(reason, result.stderr)
        for body, reason in [(b'---\nname: missing\n', 'missing ... header terminator'),
                             (b'...\n\xff\n', 'UTF-8'),
                             (b'...\n' + '词\tci\n'.encode() * 10001, 'exceeds 10000 entries'),
                             (b'...\n#' + b'x' * (8 * 1024 * 1024 - 4), 'exceeds 8 MiB')]:
            with self.subTest(reason=reason):
                self.user.write_bytes(body)
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(self.user), result.stderr)
                self.assertIn(reason, result.stderr)


if __name__ == '__main__':
    unittest.main()
