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

    def install(*args, **changes):
        return subprocess.run([fixture / 'packaging/install-local.sh', *args], env=dict(env, **changes), capture_output=True, text=True)

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
    assert 'use --upgrade' in install().stderr
    result = install('--rollback')
    assert result.returncode != 0 and 'no previous installation' in result.stderr

    # Upgrades replace only verified assets; custom files and private state stay current.
    previous = prefix / 'share/.yanliu-previous'
    custom = share / 'custom/words.dict.yaml'
    custom.parent.mkdir()
    custom.write_text('user-maintained dictionary\n')
    nested_custom = share / 'models/user-notes.txt'
    nested_custom.write_text('notes in a managed directory\n')
    state = temp / 'private-state/learning.json'
    state.parent.mkdir()
    state.write_text('fixture private learning\n')
    (share / 'custom-link').symlink_to(state)
    unchanged = {path: path.read_bytes() for path in (config_file, installed[2], installed[3], state)}
    first_manifest = (share / 'installation.json').read_bytes()
    (fixture / 'target/release/yanliu-ibus').write_text('updated fixture binary\n')
    result = install('--upgrade')
    assert result.returncode == 0, result.stderr
    assert 'Reload the Yanliu IBus service' in result.stdout
    assert (share / 'bin/yanliu-ibus').read_text() == 'updated fixture binary\n'
    assert (previous / 'installation.json').read_bytes() == first_manifest
    second_manifest = (share / 'installation.json').read_bytes()

    def assert_preserved():
        assert custom.read_text() == 'user-maintained dictionary\n'
        assert nested_custom.read_text() == 'notes in a managed directory\n'
        assert (share / 'custom-link').readlink() == state
        assert not (previous / 'custom').exists() and not (previous / 'custom-link').is_symlink()
        assert all(path.read_bytes() == text for path, text in unchanged.items())

    assert_preserved()
    result = install('--rollback')
    assert result.returncode == 0, result.stderr
    assert (share / 'installation.json').read_bytes() == first_manifest
    assert (previous / 'installation.json').read_bytes() == second_manifest
    assert_preserved()
    assert install('--upgrade').returncode == 0
    assert (share / 'installation.json').read_bytes() == second_manifest
    assert_preserved()

    # A failure after activation restores both versions and already-moved custom files.
    failure = '''import runpy, sys
from pathlib import Path
original = Path.rename
failed = False
def rename(path, target):
    global failed
    if path.name == 'custom-link' and not failed:
        failed = True
        raise OSError('fixture late activation failure')
    return original(path, target)
Path.rename = rename
sys.argv = [sys.argv[1], sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    for action in ('--upgrade', '--rollback'):
        result = subprocess.run(['/usr/bin/python3', '-c', failure, str(fixture / 'packaging/install_assets.py'), action],
                                env=env, capture_output=True, text=True)
        assert result.returncode != 0 and 'fixture late activation failure' in result.stderr, result.stderr
        assert (share / 'installation.json').read_bytes() == second_manifest
        assert (previous / 'installation.json').read_bytes() == first_manifest
        assert_preserved()
        assert not list((prefix / 'share').glob('.yanliu-install-*')), 'complete recovery should clean staging'

    # Failed recovery must leave every version and custom item available for manual repair.
    recovery_failure = '''import runpy, sys
from pathlib import Path
original = Path.rename
failed = False
point = sys.argv[3]
def rename(path, target):
    global failed
    target = Path(target)
    if path.name == 'custom-link' and not failed:
        failed = True
        raise OSError('fixture initial activation failure')
    if failed and (
        (point == 'rollback-stage' and path.name == 'yanliu' and path.parent.name.startswith('.yanliu-install-')
         and target.name == '.yanliu-previous')
        or (point == 'upgrade-older' and path.name == 'older')
        or (point == 'unknown-reverse' and path.name == 'custom' and path.parent.name == 'yanliu')
        or (point == 'current-directory' and path.name == '.yanliu-previous' and target.name == 'yanliu')
    ):
        raise OSError('fixture recovery rename failure')
    return original(path, target)
Path.rename = rename
sys.argv = [sys.argv[1], sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    built = fixture / 'target/release/yanliu-ibus'
    original_binary = built.read_bytes()
    for point in ('rollback-stage', 'upgrade-older', 'unknown-reverse', 'current-directory'):
        scenario = temp / f'recovery-{point}'
        scenario_prefix, scenario_config = scenario / 'local', scenario / 'config'
        changes = dict(YANLIU_PREFIX=str(scenario_prefix), XDG_CONFIG_HOME=str(scenario_config))
        built.write_text('fixture recovery old version\n')
        result = install(**changes)
        assert result.returncode == 0, result.stderr
        scenario_share = scenario_prefix / 'share/yanliu'
        scenario_previous = scenario_prefix / 'share/.yanliu-previous'
        old_version = (scenario_share / 'installation.json').read_bytes()
        built.write_text('fixture recovery current version\n')
        result = install('--upgrade', **changes)
        assert result.returncode == 0, result.stderr
        current_version = (scenario_share / 'installation.json').read_bytes()
        (scenario_share / 'custom').mkdir()
        (scenario_share / 'custom/words.dict.yaml').write_text('recovery user dictionary\n')
        (scenario_share / 'custom-link').symlink_to(state)
        scenario_config_file = scenario_config / 'yanliu/config.json'
        config_before = scenario_config_file.read_bytes()
        built.write_text('fixture recovery proposed version\n')
        action = '--rollback' if point == 'rollback-stage' else '--upgrade'
        result = subprocess.run(['/usr/bin/python3', '-c', recovery_failure,
                                 str(fixture / 'packaging/install_assets.py'), action, point],
                                env=dict(env, **changes), capture_output=True, text=True)
        assert result.returncode != 0 and 'fixture initial activation failure' in result.stderr, result.stderr
        assert 'fixture recovery rename failure' in result.stderr, result.stderr
        marker = 'recovery assets preserved in '
        assert marker in result.stderr, result.stderr
        retained = Path(result.stderr.split(marker, 1)[1].splitlines()[0])
        assert retained.parent == scenario_share.parent and retained.is_dir(), 'recovery staging must survive process exit'
        versions = [scenario_share, scenario_previous, retained / 'yanliu', retained / 'older']
        manifests = [(path / 'installation.json').read_bytes() for path in versions
                     if (path / 'installation.json').is_file()]
        assert old_version in manifests and current_version in manifests, 'existing assets must remain recoverable'
        for version in versions:
            if (version / 'installation.json').is_file():
                files = json.loads((version / 'installation.json').read_text())['files']
                assert all(hashlib.sha256((version / name).read_bytes()).hexdigest() == expected
                           for name, expected in files.items()), 'retained versions need their actual managed assets'
        custom_files = [path / 'custom/words.dict.yaml' for path in versions
                        if (path / 'custom/words.dict.yaml').is_file()]
        assert len(custom_files) == 1 and custom_files[0].read_text() == 'recovery user dictionary\n'
        links = [path / 'custom-link' for path in versions if (path / 'custom-link').is_symlink()]
        assert len(links) == 1 and links[0].readlink() == state
        assert scenario_config_file.read_bytes() == config_before
        assert state.read_bytes() == unchanged[state]
    built.write_bytes(original_binary)

    result = install('--upgrade', YANLIU_MODEL_DIR=str(temp / 'missing-model'))
    assert result.returncode != 0 and 'missing-model' in result.stderr
    assert (share / 'installation.json').read_bytes() == second_manifest
    assert (previous / 'installation.json').read_bytes() == first_manifest
    assert_preserved()
    (share / 'models/NOTICE').write_text('user model notes\n')
    (fixture / 'data/models/NOTICE').write_text('new upstream attribution\n')
    result = install('--upgrade')
    assert result.returncode != 0 and 'conflict with an unmanaged file' in result.stderr
    assert (share / 'models/NOTICE').read_text() == 'user model notes\n'
    (share / 'models/NOTICE').unlink()
    (fixture / 'data/models/NOTICE').unlink()
    (previous / 'keep-custom.txt').write_text('must not delete\n')
    for action in ('--upgrade', '--rollback'):
        result = install(action)
        assert result.returncode != 0 and 'unmanaged files' in result.stderr
        assert (previous / 'keep-custom.txt').read_text() == 'must not delete\n'
    (previous / 'keep-custom.txt').unlink()

    # The historical launcher and manifest stay intact through upgrade, rollback and upgrade.
    launcher = prefix / 'bin/yanliu-ibus'
    legacy_launcher = f'#!/bin/sh\nset -eu\nexec "{share}/bin/yanliu-ibus" "$@"\n'.encode()
    launcher.unlink()
    launcher.write_bytes(legacy_launcher)
    launcher.chmod(0o755)
    before_launcher_check = (share / 'installation.json').read_bytes()
    foreign_launcher = b'#!/bin/sh\nset -eu\nexec "/usr/bin/other-input-method" "$@"\n'
    for invalid in (foreign_launcher, legacy_launcher.replace(b'set -eu\n', b'set -eu\necho third-party\n')):
        launcher.write_bytes(invalid)
        result = install('--upgrade')
        assert result.returncode != 0 and 'does not point to the managed service' in result.stderr
        assert launcher.read_bytes() == invalid
        assert (share / 'installation.json').read_bytes() == before_launcher_check
        assert 'third-party' not in result.stdout, 'installer must not execute the launcher'
    launcher.write_bytes(legacy_launcher)
    record = json.loads(second_manifest)
    provenance = share / 'models/provenance.json'
    provenance.write_text('{"fixture":"historical"}\n')
    record['files']['models/provenance.json'] = hashlib.sha256(provenance.read_bytes()).hexdigest()
    record['selected_model'] = record.pop('model')['source_path']
    record['model_provenance'] = 'models/provenance.json'
    (share / 'installation.json').write_text(json.dumps(record))
    result = install('--upgrade')
    assert result.returncode == 0, result.stderr
    assert json.loads((previous / 'installation.json').read_text()) == record
    assert launcher.read_bytes() == legacy_launcher and launcher.stat().st_mode & 0o7777 == 0o755
    for action in ('--rollback', '--upgrade'):
        result = install(action)
        assert result.returncode == 0, result.stderr
        assert launcher.read_bytes() == legacy_launcher and launcher.stat().st_mode & 0o7777 == 0o755
    assert json.loads((previous / 'installation.json').read_text()) == record
    assert_preserved()
    damaged = share / 'bin/yanliu-ibus'
    damaged.write_text('locally modified managed binary\n')
    result = install('--upgrade')
    assert result.returncode != 0 and 'installed SHA-256' in result.stderr
    assert damaged.read_text() == 'locally modified managed binary\n'
    damaged.write_bytes((fixture / 'target/release/yanliu-ibus').read_bytes())
    current_manifest = (share / 'installation.json').read_bytes()
    unsupported = json.loads(current_manifest)
    unsupported['obsolete_manifest_field'] = True
    (share / 'installation.json').write_text(json.dumps(unsupported))
    result = install('--upgrade')
    assert result.returncode != 0 and 'unsupported installation manifest fields' in result.stderr
    (share / 'installation.json').write_bytes(current_manifest)
    config_file.write_text(json.dumps({'data_dir': str(share), 'obsolete_option': True}))
    result = install('--upgrade')
    assert result.returncode != 0 and 'unknown config fields' in result.stderr
    config_file.write_bytes(unchanged[config_file])
    assert_preserved()
    # Uninstall's historical contract removes the installed asset tree; move test custom files out first.
    custom.parent.rename(temp / 'saved-custom')
    nested_custom.rename(temp / 'saved-model-notes.txt')
    (share / 'custom-link').unlink()

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
    npu_config = config_file.read_bytes()
    npu_manifest = (share / 'installation.json').read_bytes()
    result = install('--rollback')
    assert result.returncode != 0 and 'previous assets do not support configured device npu' in result.stderr
    assert (share / 'installation.json').read_bytes() == npu_manifest
    result = install('--upgrade', **npu)
    assert result.returncode == 0, result.stderr
    assert (previous / 'installation.json').read_bytes() == npu_manifest
    (fixture / 'cargo-call').unlink()
    result = install('--rollback')
    assert result.returncode == 0, result.stderr
    assert not (fixture / 'cargo-call').exists(), 'rollback must not compile'
    assert (share / 'installation.json').read_bytes() == npu_manifest
    assert config_file.read_bytes() == npu_config
    uninstall()
    assert_refused(install(), "selects device 'npu'")
    print('PASS: CPU default/selected model and NPU installation layout, component XML, config preservation and device agreement, '
          'selected model and IR hashes, complete licenses and attribution hashes, runtime provenance, '
          'verified upgrade/rollback, custom-file and private-state preservation, late-failure recovery, '
          'overwrite refusal and reversible uninstall')
