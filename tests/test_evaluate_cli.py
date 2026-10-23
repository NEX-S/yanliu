"""Exercise real JSONL input and isolated candidate checking, without desktop IBus."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('YANLIU_BIN', ROOT / 'target/debug/yanliu')).resolve()
MODEL = ROOT / 'data/models/sentence-model.safetensors'


class EvaluateCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='yanliu-check-cli-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        dictionaries = self.root / 'dictionaries'
        dictionaries.mkdir()
        (dictionaries / '8105.dict.yaml').write_text(
            '...\n你\tni\t100\n年\tnian\t200\n好\thao\t100\n你好\tni hao\t1000\n现\txian\t1000\n西安\txi an\t100\n')
        for name in ['base.dict.yaml', 'ext.dict.yaml']:
            (dictionaries / name).write_text('...\n')
        self.command = [str(BINARY), '--dictionary-dir', str(dictionaries)]
        self.environment = dict(os.environ, XDG_STATE_HOME=str(self.root / 'unused-state'))

    def run_cli(self, arguments, cases):
        completed = subprocess.run(self.command + arguments, env=self.environment,
            input=''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in cases),
            text=True, capture_output=True, timeout=30)
        self.assertFalse((self.root / 'unused-state').exists())
        return completed

    def cases(self):
        return [
            {'id': 'word', 'input': 'nihao', 'gold': '你好', 'bucket': 'all',
             'source': 'public-synthetic', 'metadata': {'domain': 'general'}},
            {'id': 'context', 'input': 'xian', 'context': '我去', 'gold': '西安'},
            {'id': 'completion', 'input': 'ni', 'gold': '年', 'bucket': 'short'},
            {'id': 'missing', 'input': 'zzzz', 'golds': ['数据库']},
        ]

    def test_stdin_keeps_all_cases_and_does_not_count_completion(self):
        result = self.run_cli(['evaluate', '--data', '-'], self.cases())
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['format'], 'yanliu-candidate-check-v1')
        self.assertEqual(report['buckets']['all']['n'], 4)
        self.assertEqual(report['buckets']['all']['recall'], 2)
        self.assertEqual(report['buckets']['short']['n'], 1)
        self.assertEqual([row['id'] for row in report['rows']], ['word', 'context', 'completion', 'missing'])
        rows = {row['id']: row for row in report['rows']}
        self.assertEqual(rows['word']['rank'], 1)
        self.assertIsNone(rows['completion']['rank'])
        self.assertTrue(any(c['text'] == '年' and c['source'] == 'Completion'
                            for c in rows['completion']['candidates']))
        self.assertIsNone(rows['missing']['rank'])
        for row in report['rows']:
            self.assertEqual(row['rank'], row['static_rank'])

    def test_model_check_reorders_only_actual_candidates_and_writes_privately(self):
        static = {row['id']: row for row in json.loads(
            self.run_cli(['evaluate', '--data', '-'], self.cases()).stdout)['rows']}
        output = self.root / 'check.json'
        result = self.run_cli(['--model', str(MODEL), 'evaluate', '--data', '-',
                               '--output', str(output)], self.cases())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        report = json.loads(output.read_text())
        self.assertEqual(json.loads(result.stdout), report['buckets'])
        self.assertEqual(report['buckets']['all']['n'], 4)
        self.assertEqual(report['buckets']['all']['recall'], 2)
        self.assertEqual(report['rows'][1]['rank'], 1)
        self.assertEqual(report['rows'][1]['static_rank'], 2)
        self.assertEqual(report['buckets']['all']['rescues'], 1)
        key = lambda candidate: json.dumps(candidate, sort_keys=True)
        for row in report['rows']:
            self.assertEqual(sorted(row['candidates'], key=key),
                             sorted(static[row['id']]['candidates'], key=key))
        again = self.run_cli(['evaluate', '--data', '-', '--output', str(output)], self.cases())
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(json.loads(output.read_text()), report)

    def test_invalid_jsonl_reports_the_line_without_output(self):
        output = self.root / 'invalid.json'
        for case in [{'input': '你', 'gold': '你'}, {'input': 'ni', 'gold': ''}, {'input': 'ni'}]:
            result = self.run_cli(['evaluate', '--data', '-', '--output', str(output)],
                                  [self.cases()[0], case])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('-:2:', result.stderr)
            self.assertFalse(output.exists())

    def test_neural_recall_reports_the_actual_scoring_menu_before_promotion(self):
        (self.root / 'dictionaries/8105.dict.yaml').write_text(
            '...\n西\txi\t1\n安\tan\t90\n'
            + ''.join(f'{char}\txi\t1000\n' for char in '喜戏系夕细息习稀希吸席袭悉昔惜熄'))
        cases = [{'input': 'xian', 'context': '我去', 'gold': '西安'}]
        static = self.run_cli(['evaluate', '--data', '-'], cases)
        self.assertEqual(static.returncode, 0, static.stderr)
        original = json.loads(static.stdout)['rows'][0]
        self.assertIsNone(original['rank'])
        self.assertIsNone(original['scoring'])
        result = self.run_cli(['--model', str(MODEL), 'evaluate', '--data', '-'], cases)
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(result.stdout)['rows'][0]
        self.assertEqual(row['rank'], 1)
        self.assertIsNone(row['static_rank'])
        scoring = row['scoring']
        index = next(i for i, candidate in enumerate(scoring['candidates'])
                     if candidate['text'] == '西安')
        self.assertEqual(scoring['promotion']['index'], index)
        self.assertIn(index, [score['index'] for score in scoring['promotion']['scores']])
        self.assertEqual(scoring['candidates'][index], row['candidates'][0])
        self.assertEqual(row['candidates'][0]['source'], 'Composed')
        self.assertEqual(row['candidates'][0]['consumed'], 4)
        for candidate in original['candidates']:
            self.assertIn(candidate, row['candidates'])


if __name__ == '__main__':
    unittest.main()
