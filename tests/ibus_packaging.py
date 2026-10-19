#!/usr/bin/python3
"""Install and uninstall a fixture tree with a stub cargo; the real binary is checked by scripts/test_ibus.py."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import gi

gi.require_version('IBus', '1.0')
from gi.repository import IBus

ROOT = Path(__file__).resolve().parents[1]

with tempfile.TemporaryDirectory(prefix='yanliu-packaging-') as temp:
    temp = Path(temp)
    fixture = temp / 'fixture'
    shutil.copytree(ROOT / 'packaging', fixture / 'packaging', ignore=shutil.ignore_patterns('__pycache__'))
    (fixture / 'target/release').mkdir(parents=True)
    (fixture / 'target/release/yanliu-ibus').write_text('fixture binary\n')
    for name in ('dictionaries/base.dict.yaml', 'eval/cases.jsonl', 'THIRD_PARTY.md', 'manifest.json',
                 'models/sentence-model.safetensors', 'models/sentence-model.xml', 'models/sentence-model.openvino.json'):
        (fixture / 'data' / name).parent.mkdir(parents=True, exist_ok=True)
        (fixture / 'data' / name).write_text(f'fixture {name}\n')
    license_files = ('chinese-ime-lm/LICENSE', 'chinese-ime-training/NOTICE')
    for name in license_files:
        (fixture / 'vendor' / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'vendor' / name, fixture / 'vendor' / name)
    (temp / 'stub-bin').mkdir()
    (temp / 'stub-bin/cargo').write_text('#!/bin/sh\nprintf \'%s\\n%s\\n\' "$RUSTFLAGS" "$*" > cargo-call\n')
    (temp / 'stub-bin/cargo').chmod(0o755)
    prefix, config = temp / 'local', temp / 'config home/砚流'
    config_file = config / 'yanliu/config.json'
    installed = [prefix / 'share/yanliu', prefix / 'bin/yanliu-ibus', prefix / 'share/ibus/component/yanliu.xml',
                 config / 'environment.d/60-yanliu.conf']
    env = dict(os.environ, YANLIU_PREFIX=str(prefix), XDG_CONFIG_HOME=str(config),
               PATH=f'{temp}/stub-bin:{os.environ["PATH"]}', IBUS_COMPONENT_PATH='/existing/ibus/component')

    def install(**changes):
        return subprocess.run([fixture / 'packaging/install-local.sh'], env=dict(env, **changes), capture_output=True, text=True)

    def uninstall():
        subprocess.run([fixture / 'packaging/uninstall-local.sh'], env=env, check=True, capture_output=True)

    def assert_refused(result, *reasons, config_home=config):
        assert result.returncode != 0 and all(reason in result.stderr for reason in reasons), result.stderr
        assert not any(path.exists() or path.is_symlink() for path in installed[:3])
        assert not (config_home / 'environment.d/60-yanliu.conf').exists()

    result = install()
    assert result.returncode == 0, result.stderr
    rustflags, cargo = (fixture / 'cargo-call').read_text().splitlines()
    assert rustflags.endswith('-C target-cpu=native') and cargo == 'build --locked --release --features ibus --bin yanliu-ibus'
    share = prefix / 'share/yanliu'
    assert (prefix / 'bin/yanliu-ibus').resolve() == share / 'bin/yanliu-ibus'
    assert sorted(str(path.relative_to(share)) for path in share.rglob('*') if path.is_file()) == [
        'THIRD_PARTY.md', 'bin/yanliu-ibus', 'dictionaries/base.dict.yaml', 'installation.json',
        'licenses/chinese-ime-lm/LICENSE', 'licenses/chinese-ime-training/NOTICE',
        'models/sentence-model.safetensors', 'source-manifest.json']
    record = json.loads((share / 'installation.json').read_text())
    assert record['model'] == {'source_path': str(fixture / 'data/models/sentence-model.safetensors'),
                               'sha256': hashlib.sha256((share / 'models/sentence-model.safetensors').read_bytes()).hexdigest()}
    assert record['files'] == {name: hashlib.sha256((share / name).read_bytes()).hexdigest()
                               for name in record['files']} and len(record['files']) == 7
    for name in license_files:
        original = (ROOT / 'vendor' / name).read_bytes()
        assert (share / 'licenses' / name).read_bytes() == original
        assert record['files'][f'licenses/{name}'] == hashlib.sha256(original).hexdigest()
    component = IBus.Component.new_from_file(str(prefix / 'share/ibus/component/yanliu.xml'))
    assert component.get_name() == 'org.freedesktop.IBus.Yanliu'
    assert [(e.get_name(), e.get_language()) for e in component.get_engines()] == [('yanliu', 'zh_CN')]
    assert shlex.split(component.get_exec()) == [str(prefix / 'bin/yanliu-ibus'), '--ibus', '--config', str(config_file)]
    assert json.loads(config_file.read_text()) == {'data_dir': str(share)}
    assert installed[3].read_text() == f'IBUS_COMPONENT_PATH={prefix}/share/ibus/component:/existing/ibus/component\n'
    assert 'uninstall first' in install().stderr

    (prefix / 'bin/other-input-method').write_text('preserve\n')
    preserved = json.dumps({'data_dir': str(share), 'learning_enabled': False, 'user_dictionary': 'words.dict.yaml',
                            'contextual_learning': True})
    config_file.write_text(preserved)
    uninstall()
    assert not any(path.exists() or path.is_symlink() for path in installed)
    assert (prefix / 'bin/other-input-method').read_text() == 'preserve\n'
    assert install().returncode == 0 and config_file.read_text() == preserved, 'existing config is kept verbatim'
    uninstall()

    for invalid in ('{', '[]'):
        config_file.write_text(invalid)
        assert_refused(install(), str(config_file))
        assert config_file.read_text() == invalid
    config_file.write_text('{}')
    assert_refused(install(YANLIU_DEVICE='npu', YANLIU_MODEL_DIR=str(temp),
                           YANLIU_ACCELERATOR_RUNTIME_DIR=str(temp)), 'YANLIU_DEVICE')
    assert_refused(install(YANLIU_ACCELERATOR_RUNTIME_DIR=str(temp)), 'YANLIU_ACCELERATOR_RUNTIME_DIR')
    config_file.unlink()

    # A late failure rolls back new files, including a new config, and keeps unrelated ones.
    blocker = temp / 'config-blocker'
    blocker.write_text('unrelated file')
    assert_refused(install(XDG_CONFIG_HOME=str(blocker)), config_home=blocker)
    assert blocker.read_text() == 'unrelated file'
    late = temp / 'late-config'
    late.mkdir()
    (late / 'environment.d').write_text('unrelated blocker')
    assert_refused(install(XDG_CONFIG_HOME=str(late)), config_home=late)
    assert not (late / 'yanliu/config.json').exists() and (late / 'environment.d').read_text() == 'unrelated blocker'
    (late / 'yanliu/config.json').write_text('{"learning_enabled":false}')
    assert_refused(install(XDG_CONFIG_HOME=str(late)), config_home=late)
    assert (late / 'yanliu/config.json').read_text() == '{"learning_enabled":false}'

    ranker, runtime = temp / 'ranker', temp / 'runtime'
    for suffix in ('safetensors', 'xml', 'bin'):
        (ranker / f'sentence-model.{suffix}').parent.mkdir(exist_ok=True)
        (ranker / f'sentence-model.{suffix}').write_text(f'exported {suffix}\n')
    (ranker / 'sentence-model.openvino.json').write_text(json.dumps({'format': 'yanliu-ranker-ir-v1', **{
        field: hashlib.sha256((ranker / f'sentence-model.{suffix}').read_bytes()).hexdigest()
        for suffix, field in (('safetensors', 'source_sha256'), ('xml', 'xml_sha256'), ('bin', 'bin_sha256'))}}))
    model_notices = {'LICENSE': 'selected model license\n', 'NOTICE': 'selected model attribution\n',
                     'README.md': 'selected model card\n', 'provenance.json': '{"fixture":true}\n'}
    for name, text in model_notices.items():
        (ranker / name).write_text(text)
    result = install(YANLIU_MODEL_DIR=str(ranker))
    assert result.returncode == 0, result.stderr
    assert (share / 'models/sentence-model.safetensors').read_text() == 'exported safetensors\n'
    assert not (share / 'runtime').exists() and not (share / 'models/sentence-model.xml').exists()
    record = json.loads((share / 'installation.json').read_text())
    assert record['model']['source_path'] == str(ranker / 'sentence-model.safetensors')
    for name, text in model_notices.items():
        assert (share / 'models' / name).read_text() == text
        assert record['files']['models/' + name] == hashlib.sha256(text.encode()).hexdigest()
    uninstall()
    config_file.unlink()
    names = ['libopenvino.so.2600', 'libopenvino_intel_gpu_plugin.so', 'libopenvino_intel_npu_plugin.so',
             'libopenvino_intel_npu_compiler.so', 'licenses/LICENSE']
    for name in names:
        (runtime / name).parent.mkdir(parents=True, exist_ok=True)
        (runtime / name).write_text(f'official {name}\n')
    (runtime / 'libopenvino.so').symlink_to('libopenvino.so.2600')
    source = {'url': 'https://example.invalid/runtime.tar.gz', 'sha256': '0' * 64}
    (runtime / 'runtime-source.json').write_text(json.dumps({'format': 'yanliu-runtime-v1', 'source': source, 'files': {
        name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in names + ['libopenvino.so']}}))
    npu = dict(YANLIU_DEVICE='npu', YANLIU_MODEL_DIR=str(ranker), YANLIU_ACCELERATOR_RUNTIME_DIR=str(runtime))
    for suffix in ('safetensors', 'xml', 'bin'):
        model_file = ranker / f'sentence-model.{suffix}'
        original = model_file.read_bytes()
        model_file.write_bytes(b'modified\n')
        assert_refused(install(**npu), str(model_file), 'SHA-256 does not match ranker metadata')
        assert not config_file.exists()
        model_file.write_bytes(original)
    (runtime / 'libopenvino_intel_npu_plugin.so').write_text('modified\n')
    assert_refused(install(**npu), 'libopenvino_intel_npu_plugin.so')
    (runtime / 'libopenvino_intel_npu_plugin.so').write_text('official libopenvino_intel_npu_plugin.so\n')
    result = install(**npu)
    assert result.returncode == 0, result.stderr
    record = json.loads((share / 'installation.json').read_text())
    assert record['model'] == {'source_path': str(ranker / 'sentence-model.safetensors'),
                               'sha256': hashlib.sha256((share / 'models/sentence-model.safetensors').read_bytes()).hexdigest()}
    for name in license_files:
        original = (ROOT / 'vendor' / name).read_bytes()
        assert (share / 'licenses' / name).read_bytes() == original
        assert record['files'][f'licenses/{name}'] == hashlib.sha256(original).hexdigest()
    assert (fixture / 'cargo-call').read_text().splitlines()[1] == 'build --locked --release --features ibus,accelerator --bin yanliu-ibus'
    assert {path.name for path in (share / 'models').iterdir()} == {
        'sentence-model.bin', 'sentence-model.openvino.json', 'sentence-model.safetensors', 'sentence-model.xml',
        *model_notices}
    for name, text in model_notices.items():
        assert (share / 'models' / name).read_text() == text
        assert record['files']['models/' + name] == hashlib.sha256(text.encode()).hexdigest()
    assert (share / 'models/sentence-model.safetensors').read_text() == 'exported safetensors\n'
    assert (share / 'models/sentence-model.xml').read_text() == 'exported xml\n'
    installed_runtime = json.loads((share / 'runtime/runtime-source.json').read_text())
    assert installed_runtime['source'] == source
    assert sorted(installed_runtime['files']) == ['libopenvino.so', 'libopenvino.so.2600', 'libopenvino_intel_npu_compiler.so',
                                                  'libopenvino_intel_npu_plugin.so', 'licenses/LICENSE']
    assert {str(path.relative_to(share / 'runtime')) for path in (share / 'runtime').rglob('*')
            if not path.is_dir()} == {*installed_runtime['files'], 'runtime-source.json'}
    assert os.readlink(share / 'runtime/libopenvino.so') == 'libopenvino.so.2600'
    assert json.loads(config_file.read_text()) == {'data_dir': str(share), 'device': 'npu', 'runtime_dir': str(share / 'runtime')}
    uninstall()
    assert_refused(install(), "selects device 'npu'")
    print('PASS: CPU default/selected model and NPU installation layout, component XML, config preservation and device agreement, '
          'selected model and IR hashes, complete licenses and attribution hashes, runtime provenance, '
          'rollback, overwrite refusal and reversible uninstall')
