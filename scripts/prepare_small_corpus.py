#!/usr/bin/env python3
"""Prepare the fixed mixed corpus for continued 4M/7M training from their real parents."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import unicodedata

import build_ranking_corpus as ranking
import prepare_general_corpus as public

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'results/retrain-small-20261104/corpus'
RANKING = ROOT / 'results/retrain-small-20261104/evaluation-coverage'
RUN = ROOT / 'results/retrain-small-adapt-20261104'
BUDGETS = {'lccc': 15_000_000, 'wiki': 25_000_000, 'cqia': 8_400_000,
           'cped': None, 'tnews': 1_000_000, 'edu': 15_000_000}
PARENTS = {
    '4m': (ROOT / 'results/retrain-20261021/compact-ranking/epoch01-f32.safetensors',
           'be480d9aec0e9de4c48f35899756027d4d8a347ffa5d92656965eaa88f8d5afc'),
    '7m': (ROOT / 'results/general-model-20261021/selected-ranking/epoch01-f32.safetensors',
           '8d01642e7a669280e52314e70a8982c490042aa9d82d8ed87aa0c939086dcdee')}


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def query_key(case):
    return ranking.hashed(*(''.join(c for c in unicodedata.normalize('NFKC', value).casefold() if c.isalnum())
                            for value in (case['input'], case.get('context', ''))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    inputs_path = RUN / 'inputs.json'
    inputs = json.loads(inputs_path.read_text())
    assert inputs['status'] == 'complete'
    source_paths = defaultdict(list)
    boundaries = [inputs_path, PREVIOUS / 'documents.jsonl', PREVIOUS / 'recipe.json',
                  *(RANKING / f'ranking-{split}.jsonl' for split in ranking.SPLITS)]
    for source in inputs['files']:
        path = Path(source['path'])
        assert source['status'] == 'verified' and public.digest(path) == source['expected_sha256']
        source_paths[source['source']].append(path)
        boundaries.append(path)
    parents = {}
    for name, (path, expected) in PARENTS.items():
        assert public.digest(path) == expected
        with path.open('rb') as stream:
            metadata = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))['__metadata__']
        parents[name] = {'path': str(path), 'sha256': expected, 'config': json.loads(metadata['config']),
                         'vocab': json.loads(metadata['vocab'])}
        assert parents[name]['config']['vocab'] == len(parents[name]['vocab'])
        boundaries.append(path)
    eval_files = sorted((ROOT / 'data/eval').glob('*.jsonl'))
    exclusion_files = [*eval_files, RANKING / 'ranking-dev.jsonl', RANKING / 'ranking-holdout.jsonl']
    excluded, exclusion_records = public.evaluation_exclusions(exclusion_files)
    exclusion_pattern = re.compile('|'.join(re.escape(text) for text in sorted(excluded)))
    boundaries.extend(eval_files)
    input_hashes = {str(path): public.digest(path) for path in boundaries}
    stats = defaultdict(Counter)
    counts = defaultdict(Counter)
    selected_groups = defaultdict(lambda: defaultdict(set))
    frequencies = defaultdict(Counter)
    group_splits = {}
    original_paragraphs = defaultdict(Counter)
    original_groups = defaultdict(lambda: defaultdict(set))
    reserved = public.open_dedup(args.output / 'reserved.sqlite3')
    seen = public.open_dedup(args.output / 'seen.sqlite3')
    for row in rows(PREVIOUS / 'documents.jsonl'):
        original_paragraphs[row['source']][row['split']] += 1
        original_groups[row['source']][row['split']].add(row['split_group'])
        if row['split'] != 'train':
            normalized = public.canonical(row['text'])
            keys = public.near_keys(normalized)
            public.remember_text(reserved, normalized, keys)
            public.remember_text(seen, normalized, keys)
            stats[row['source']]['reserved_original_' + row['split']] += 1
    reserved.commit()
    seen.commit()
    print(json.dumps({'stage': 'reserved_original_evaluation', 'stats': dict(stats)}), flush=True)
    stage = sqlite3.connect(args.output / 'staging.sqlite3')
    stage.execute('PRAGMA journal_mode=OFF')
    stage.execute('PRAGMA synchronous=OFF')
    stage.execute('CREATE TABLE docs(source TEXT, grp TEXT, split_group TEXT, split TEXT, priority BLOB, record BLOB, paragraphs TEXT, PRIMARY KEY(source,grp,record)) WITHOUT ROWID')
    for source in ('lccc', 'wiki', 'cqia', 'cped', 'tnews'):
        for group, split_group, split, text in public.source_records(source, source_paths[source]):
            stats[source]['raw_records'] += 1
            if source == 'cped':
                group = split_group + ':' + group
            paragraphs = text.split('\n\n') if source in ('lccc', 'cped') and text else list(public.clean_paragraphs(text))
            if not paragraphs:
                stats[source]['empty_after_cleaning'] += 1
                continue
            before = stage.total_changes
            stage.execute('INSERT OR IGNORE INTO docs VALUES(?,?,?,?,?,?,?)',
                          (source, group, split_group, split, public.fingerprint(source + '\0' + group),
                           public.fingerprint(public.canonical(text)), json.dumps(paragraphs, ensure_ascii=False)))
            if before == stage.total_changes:
                stats[source]['identical_record_in_group'] += 1
            if stats[source]['raw_records'] % 50000 == 0:
                stage.commit()
                print(json.dumps({'stage': 'source_staging', 'source': source,
                                  'records': stats[source]['raw_records']}), flush=True)
        stage.commit()
        print(json.dumps({'stage': 'source_staged', 'source': source, 'counts': stats[source]}), flush=True)
    stage.execute('CREATE INDEX source_order ON docs(source,split,priority,grp)')
    stage.commit()
    previous_stage = PREVIOUS / 'staging.sqlite3'
    previous = sqlite3.connect(f'file:{previous_stage}?mode=ro', uri=True)
    old_cqia = dict(previous.execute("SELECT DISTINCT grp,split FROM docs WHERE source='cqia' AND split!='train'"))
    previous.close()
    cqia_owners = {}
    for path in source_paths['cqia']:
        human_rows = (row for row in rows(path) if row.get('answer_from') == 'human')
        for row, record in zip(human_rows, public.source_records('cqia', [path]), strict=True):
            owner = old_cqia.get(ranking.hashed(row['instruction'], row['input']).hex())
            if owner:
                group = record[1]
                assert cqia_owners.setdefault(group, owner) == owner
    moved_from_train = 0
    for group, split in cqia_owners.items():
        previous_splits = {row[0] for row in stage.execute('SELECT DISTINCT split FROM docs WHERE source=? AND split_group=?', ('cqia', group))}
        moved_from_train += 'train' in previous_splits
        stage.execute('UPDATE docs SET split=? WHERE source=? AND split_group=?', (split, 'cqia', group))
    stage.commit()
    input_hashes[str(previous_stage)] = public.digest(previous_stage)
    cqia_ownership = {'original_frozen_groups': len(old_cqia), 'mapped_question_groups': len(cqia_owners),
                      'moved_from_train': moved_from_train}
    print(json.dumps({'stage': 'protected_cqia_questions', **cqia_ownership}), flush=True)
    restored_denominators = defaultdict(dict)
    for source, split, records, groups in stage.execute('SELECT source,split,COUNT(*),COUNT(DISTINCT split_group) FROM docs GROUP BY source,split'):
        restored_denominators[source][split] = {'records': records, 'source_groups': groups}
    for (raw,) in stage.execute("SELECT paragraphs FROM docs WHERE split!='train'"):
        for text in json.loads(raw):
            normalized = public.canonical(text)
            public.remember_text(reserved, normalized, public.near_keys(normalized))
    reserved.commit()
    texts = {split: (args.output / f'{split}.txt').open('x') for split in ranking.SPLITS}
    documents = (args.output / 'documents.jsonl').open('x')

    def write(row):
        split, source, text = row['split'], row['source'], row['text']
        identity = source + ':' + row['split_group']
        assert group_splits.setdefault(identity, split) == split
        texts[split].write(text + '\n')
        documents.write(json.dumps(row, ensure_ascii=False) + '\n')
        counts[source][split + '_characters'] += len(text)
        counts[source][split + '_paragraphs'] += 1
        selected_groups[split][source].add(row['split_group'])
        if split == 'train':
            frequencies[source].update(text)

    def accept(source, split, text, limit):
        if limit is not None and counts[source][split + '_characters'] + len(text) > limit:
            stats[source][split + '_budget_not_selected'] += 1
            return False
        normalized = public.canonical(text)
        if exclusion_pattern.search(normalized):
            stats[source][split + '_evaluation_fragment_excluded'] += 1
            return False
        keys = public.near_keys(normalized)
        if split == 'train':
            reason = public.duplicate_reason(reserved, normalized, keys)
            if reason:
                stats[source]['train_reserved_' + reason] += 1
                return False
        reason = public.duplicate_reason(seen, normalized, keys)
        if reason:
            stats[source][split + '_duplicate_' + reason] += 1
            return False
        public.remember_text(seen, normalized, keys)
        return True

    for row in rows(PREVIOUS / 'documents.jsonl'):
        source, split = row['source'], row['split']
        if split != 'train':
            write(row)
        elif source == 'cqia':
            stats[source]['original_train_replaced'] += 1
        elif accept(source, split, row['text'], BUDGETS.get(source)):
            write(row)
    for source in ('lccc', 'wiki', 'cqia', 'cped', 'tnews'):
        for split in ('dev', 'holdout', 'train'):
            limit = BUDGETS[source] if split == 'train' else 100_000
            records = stage.execute('SELECT grp,split_group,paragraphs FROM docs WHERE source=? AND split=? ORDER BY priority,grp,record', (source, split))
            for group, split_group, raw in records:
                for paragraph, text in enumerate(json.loads(raw)):
                    if accept(source, split, text, limit):
                        write({'source': source, 'group': group, 'split_group': split_group,
                               'paragraph': paragraph, 'split': split, 'text': text})
            seen.commit()
        print(json.dumps({'stage': 'source_selected', 'source': source, 'counts': counts[source]}), flush=True)
    for stream in texts.values():
        stream.close()
    documents.close()
    original_dev = list(rows(RANKING / 'ranking-dev.jsonl'))
    regression_path = ROOT / 'data/eval/replay-v3-test.jsonl'
    regression = list(rows(regression_path))
    assert len(original_dev) == 1609 and len(regression) == 2000
    dev = []
    for prefix, cases in (('frozen', original_dev), ('observed-replay', regression)):
        for case in cases:
            case = dict(case, id=prefix + ':' + case['id'])
            query = query_key(case).hex()
            if prefix == 'observed-replay':
                case['group_id'] = 'observed-replay-query:' + query
                case['source_group'] = 'observed-replay:' + query
            else:
                case['group_id'] = 'frozen:' + case['group_id']
            dev.append(case)
    holdout = list(rows(RANKING / 'ranking-holdout.jsonl'))
    forbidden_queries = {query_key(case) for case in [*dev, *holdout]}
    original_train = list(rows(RANKING / 'ranking-train.jsonl'))
    train = [case for case in original_train if query_key(case) not in forbidden_queries]
    removed = [case for case in original_train if query_key(case) in forbidden_queries]
    with (args.output / 'ranking-train.jsonl').open('x') as stream:
        for case in train:
            stream.write(json.dumps(case, ensure_ascii=False) + '\n')
    with (args.output / 'ranking-dev.jsonl').open('x') as stream:
        for case in dev:
            stream.write(json.dumps(case, ensure_ascii=False) + '\n')
    shutil.copyfile(RANKING / 'ranking-holdout.jsonl', args.output / 'ranking-holdout.jsonl')
    query_overlaps = {}
    rank_cases = {'train': train, 'dev': dev, 'holdout': holdout}
    for first, second in (('train', 'dev'), ('train', 'holdout'), ('dev', 'holdout')):
        query_overlaps[first + '/' + second] = len({query_key(case) for case in rank_cases[first]} &
                                                  {query_key(case) for case in rank_cases[second]})
    assert not any(query_overlaps.values()), query_overlaps
    assert len(dev) == 3609 and len(holdout) == 583
    assert public.digest(args.output / 'ranking-holdout.jsonl') == input_hashes[str(RANKING / 'ranking-holdout.jsonl')]
    for path, expected in input_hashes.items():
        assert public.digest(Path(path)) == expected, path
    lm_group_overlaps = {}
    for first, second in (('train', 'dev'), ('train', 'holdout'), ('dev', 'holdout')):
        a = {source + ':' + group for source, groups in selected_groups[first].items() for group in groups}
        b = {source + ':' + group for source, groups in selected_groups[second].items() for group in groups}
        lm_group_overlaps[first + '/' + second] = len(a & b)
    assert not any(lm_group_overlaps.values()), lm_group_overlaps
    recipe = {'private': True, 'inputs': input_hashes, 'restored_sources': inputs,
              'source_hashes': {str(path): public.digest(path) for path in
                                (Path(__file__).resolve(), ROOT / 'scripts/prepare_general_corpus.py', ROOT / 'scripts/build_ranking_corpus.py')},
              'parent_models': {name: {key: value for key, value in parent.items() if key != 'vocab'} for name, parent in parents.items()},
              'train_character_budgets': BUDGETS, 'additional_eval_character_cap': 100_000,
              'stats': dict(stats), 'selected': dict(counts),
              'source_groups': {split: {source: len(groups) for source, groups in sources.items()}
                                for split, sources in selected_groups.items()},
              'source_group_denominators': {'restored_staged': dict(restored_denominators),
                  'original_selected': {source: {split: {'paragraphs': original_paragraphs[source][split],
                                                        'source_groups': len(groups)}
                                               for split, groups in splits.items()}
                                        for source, splits in original_groups.items()}},
              'lm_source_group_overlaps': lm_group_overlaps, 'cqia_frozen_question_ownership': cqia_ownership,
              'excluded_evaluation_fragments': {'count': len(excluded), 'files': exclusion_records,
                  'rule': 'canonical context+gold fragments of at least six characters; short bare targets do not cause global substring exclusion'},
              'ranking': {'train': len(train), 'original_train': len(original_train), 'dev': len(dev), 'holdout': len(holdout),
                  'removed_train_query_cases': len(removed), 'removed_train_query_identities': len({query_key(case) for case in removed}),
                  'removed_by_source': dict(Counter(case['source_group'].split(':', 1)[0] for case in removed)),
                  'normalized_input_context_overlaps': query_overlaps,
                  'dev_groups': len({case['group_id'] for case in dev}),
                  'regression_rows': len(regression), 'regression_groups': len({query_key(case) for case in regression}),
                  'dev_status': 'contains the complete already-observed replay-v3-test as regression protection; not an independent test',
                  'holdout_status': 'frozen 583 cases protected from this continuation; independence from historical parent pretraining is not established'},
              'existing_panels': [{'path': str(ROOT / 'data/eval' / name), 'sha256': public.digest(ROOT / 'data/eval' / name)}
                                  for name in ('common-words-v1.jsonl', 'candidate-repairs-v1.jsonl')]}
    for name, parent in parents.items():
        vocab = set(parent['vocab'])
        recipe.setdefault('unknown_train_characters', {})[name] = {
            source: sum(count for char, count in frequency.items() if char not in vocab)
            for source, frequency in frequencies.items()}
    recipe['outputs'] = {path.name: public.digest(path) for path in args.output.iterdir()
                         if path.suffix in ('.txt', '.jsonl')}
    public.write_json(args.output / 'recipe.json', recipe)
    provenance = {'license': 'Mixed source conditions retained; private assistant-derived research weights',
                  'data_classification': 'private: includes explicitly authorized local Claude/Codex/Grok assistant text',
                  'attribution': 'Restored fixed general Chinese sources plus frozen assistant/synthetic/domain text; source conditions retained in recipe.json',
                  'inputs_manifest': str((args.output / 'recipe.json').resolve()),
                  'inputs_manifest_sha256': public.digest(args.output / 'recipe.json')}
    for name, parent in parents.items():
        public.write_json(args.output / f'{name}-spec.json', {'config': parent['config'], 'vocab': parent['vocab'],
                          'provenance': dict(provenance, parent_model={'path': parent['path'], 'sha256': parent['sha256']})})
    reserved.close()
    seen.close()
    stage.close()
    for name in ('reserved.sqlite3', 'seen.sqlite3', 'staging.sqlite3'):
        (args.output / name).unlink()
    print(json.dumps({'stage': 'complete', 'selected': dict(counts), 'ranking': recipe['ranking'],
                      'lm_source_group_overlaps': lm_group_overlaps}), flush=True)


if __name__ == '__main__':
    main()
