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


class LearningReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='yanliu-learning-replay-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dictionary = self.root / 'dictionary'
        self.dictionary.mkdir()
        (self.dictionary / '8105.dict.yaml').write_text(
            '...\n现\txian\t1000\n西安\txi an\t100\n你\tni\t100\n'
            '泥\tni\t90\n好\thao\t100\n年\tnian\t90\n')
        for name in ('base.dict.yaml', 'ext.dict.yaml'):
            (self.dictionary / name).write_text('...\n')
        self.model = Path(os.environ.get(
            'YANLIU_TEST_MODEL', Path(__file__).resolve().parents[1] /
            'data/models/sentence-model.safetensors')).resolve()
        self.data = self.root / 'cases.jsonl'
        self.state = self.root / 'new-state'
        self.output = self.root / 'report.json'

    def run_replay(self, cases, state=None, output=None):
        self.data.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        return subprocess.run(
            [str(BINARY), '--dictionary-dir', str(self.dictionary), '--model', str(self.model),
             'learning', 'replay', '--data', str(self.data), '--state-dir', str(state or self.state),
             '--output', str(output or self.output)],
            env={**os.environ, 'XDG_STATE_HOME': str(self.root)},
            text=True, capture_output=True, timeout=30,
        )

    def test_opposite_contexts_scores_and_reload_match_production(self):
        fixture = Path(__file__).resolve().parents[1] / 'data/eval/learning-context.jsonl'
        cases = [json.loads(line) for line in fixture.read_text().splitlines()]
        choices = [('乘火车前往', '西安'), ('此时此刻', '现')]
        result = self.run_replay(cases)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['format'], 'yanliu-learning-replay-v1')
        self.assertTrue(report['reload_preserved_snapshot'])
        self.assertTrue(report['reload_preserved_rankings'])
        self.assertEqual(report['before_reload'], report['reloaded'])
        for row, (context, expected) in zip(report['reloaded'], choices):
            self.assertEqual(row['context'], context)
            self.assertEqual(row['result']['candidates'][0]['text'], expected)
        for row in report['rows']:
            if row['selection'] is not None:
                self.assertEqual(row['selection']['remaining_input'], '')
            for step in ('before', 'after'):
                result = row[step]
                promotion = result['promotion']
                for score in promotion['scores']:
                    self.assertAlmostEqual(
                        score['total'], (score['language'] + score['dictionary']) /
                        promotion['prior_scale'] + score['personal'] + score['frequency'], places=5)
                    self.assertGreaterEqual(score['count'], 0)
                self.assertTrue(all('score_bits' in c and 'source' in c and 'consumed' in c
                                    for c in result['candidates']))
        snapshot = json.loads((self.state / 'learning.json').read_text())
        self.assertEqual(sorted((r['text'], r['count']) for r in snapshot['rows']),
                         [('现', 4), ('西安', 4)])
        self.assertEqual(len(snapshot['adapter']), 4096)
        self.assertNotIn('context', snapshot)
        for context, _ in choices:
            self.assertNotIn(context, (self.state / 'learning.json').read_text())
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)

    def test_partial_consumption_and_completion_do_not_persist_extra_text(self):
        cases = [
            {'id': 'partial', 'input': "ni'hao", 'context': '道路上的',
             'select': {'text': '泥', 'consumed': 3}},
            {'id': 'completion', 'input': 'ni', 'select': {'text': '年', 'consumed': 2}},
        ]
        result = self.run_replay(cases)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['rows'][0]['selection']['remaining_input'], 'hao')
        self.assertTrue(report['reload_preserved_rankings'])
        snapshot = json.loads((self.state / 'learning.json').read_text())
        self.assertEqual(snapshot['rows'], [{'input': "ni'", 'text': '泥', 'count': 1}])
        self.assertNotIn('道路上的', (self.state / 'learning.json').read_text())

    def test_query_only_does_not_create_a_learning_snapshot(self):
        result = self.run_replay([{'id': 'query', 'input': 'xian'}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.state / 'learning.json').exists())
        self.assertTrue(json.loads(self.output.read_text())['reload_preserved_rankings'])

    def test_segmented_choices_remember_the_complete_phrase_and_reload(self):
        result = self.run_replay([
            {'id': 'three-parts', 'input': "Ni'HaoNi", 'selects': [
                {'text': '泥', 'consumed': 3}, {'text': '好', 'consumed': 3},
                {'text': '你', 'consumed': 2}]},
            {'id': 'legacy', 'input': 'xian', 'select': {'text': '现', 'consumed': 4}},
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertIsNone(report['rows'][0]['selection'])
        self.assertEqual(report['rows'][0]['selections'], [
            {'text': '泥', 'consumed': 3, 'remaining_input': 'HaoNi'},
            {'text': '好', 'consumed': 3, 'remaining_input': 'Ni'},
            {'text': '你', 'consumed': 2, 'remaining_input': ''},
        ])
        self.assertEqual(report['rows'][1]['selection'],
                         {'text': '现', 'consumed': 4, 'remaining_input': ''})
        self.assertNotIn('selections', report['rows'][1])
        for query in (report['rows'][0]['after'], report['reloaded'][0]['result']):
            self.assertEqual(query['candidates'][0]['text'], '泥好你')
            self.assertEqual(query['candidates'][0]['consumed'], 8)
            self.assertEqual(query['candidates'][0]['source'], 'User')
        self.assertTrue(report['reload_preserved_snapshot'])
        self.assertTrue(report['reload_preserved_rankings'])
        snapshot = json.loads((self.state / 'learning.json').read_text())
        self.assertEqual(sorted((r['input'], r['text'], r['count']) for r in snapshot['rows']),
                         sorted([("Ni'", '泥', 1), ('Hao', '好', 1), ('Ni', '你', 1),
                                 ("Ni'HaoNi", '泥好你', 1), ('xian', '现', 1)]))

    def test_segmented_choice_failure_reports_the_actual_step(self):
        result = self.run_replay([{'id': 'bad-second', 'input': "ni'hao", 'selects': [
            {'text': '你', 'consumed': 3}, {'text': '不存在', 'consumed': 3}]}])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('bad-second: selection step 2: selected candidate not recalled', result.stderr)

    def test_refuses_existing_state_or_report_and_never_reads_system_state(self):
        cases = [{'id': 'query', 'input': 'xian'}]
        self.state.mkdir()
        sentinel = self.state / 'learning.json'
        sentinel.write_bytes(b'not a snapshot; must remain untouched')
        result = self.run_replay(cases)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('state directory already exists', result.stderr)
        self.assertEqual(sentinel.read_bytes(), b'not a snapshot; must remain untouched')
        self.assertFalse(self.output.exists())
        self.output.write_text('previous report')
        result = self.run_replay(cases, state=self.root / 'another-state')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_text(), 'previous report')
        self.assertFalse((self.root / 'another-state').exists())
        self.output.unlink()
        system = self.root / 'yanliu'
        for target in (system,):
            result = self.run_replay(cases, state=target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cannot use the system learning directory', result.stderr)
            self.assertFalse(target.exists())
        system.mkdir()
        (system / 'learning.json').write_text('must not be opened')
        alias = self.root / 'alias'
        alias.symlink_to(system, target_is_directory=True)
        result = self.run_replay(cases, state=alias / 'new-child')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cannot use the system learning directory', result.stderr)
        self.assertFalse((system / 'new-child').exists())
        self.assertEqual((system / 'learning.json').read_text(), 'must not be opened')
        result = self.run_replay(cases, state=self.root / 'safe-new-state', output=alias / 'new-report')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cannot use the system learning directory', result.stderr)
        self.assertFalse((system / 'new-report').exists())
        self.assertFalse((self.root / 'safe-new-state').exists())

    def test_invalid_or_unrecalled_explicit_selection_reports_failure(self):
        for case in [
            {'id': 'bad-pinyin', 'input': 'xian1'},
            {'id': 'bad-consumed', 'input': 'xian', 'select': {'text': '现', 'consumed': 5}},
            {'id': 'unknown-field', 'input': 'xian', 'action': 'erase'},
            {'id': 'exclusive', 'input': 'xian', 'select': {'text': '现', 'consumed': 4},
             'selects': [{'text': '现', 'consumed': 4}]},
            {'id': 'empty-chain', 'input': 'xian', 'selects': []},
            {'id': 'bad-second-length', 'input': "ni'hao", 'selects': [
                {'text': '你', 'consumed': 3}, {'text': '好', 'consumed': 4}]},
        ]:
            with self.subTest(case=case):
                result = self.run_replay([case])
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.state.exists())
                self.assertFalse(self.output.exists())
        result = self.run_replay([{'id': 'missing-choice', 'input': 'xian',
                                  'select': {'text': '不存在', 'consumed': 4}}])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing-choice: selected candidate not recalled', result.stderr)
        self.assertFalse((self.state / 'learning.json').exists())


if __name__ == '__main__':
    unittest.main()
