#!/usr/bin/env python3
"""Install the built IBus service, dictionaries and selected ranker for this user; run by install-local.sh."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from xml.sax.saxutils import escape

# A GPU or NPU installation leaves out the other device's plugins.
OTHER_DEVICE_FILES = {'gpu': {'libopenvino_intel_npu_plugin.so', 'libopenvino_intel_npu_compiler.so'},
                      'npu': {'libopenvino_intel_gpu_plugin.so'}}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_accelerator(model_dir, runtime_dir, device, stage):
    for suffix in ('xml', 'bin', 'openvino.json'):
        shutil.copyfile(model_dir / f'sentence-model.{suffix}', stage / f'models/sentence-model.{suffix}')
    metadata_file = stage / 'models/sentence-model.openvino.json'
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    if metadata['format'] != 'yanliu-ranker-ir-v1':
        raise ValueError(f'{model_dir}/sentence-model.openvino.json: unsupported ranker metadata')
    for suffix, field in (('safetensors', 'source_sha256'), ('xml', 'xml_sha256'), ('bin', 'bin_sha256')):
        if digest(stage / f'models/sentence-model.{suffix}') != metadata[field]:
            raise ValueError(f'{model_dir}/sentence-model.{suffix}: SHA-256 does not match ranker metadata')
    record = json.loads((runtime_dir / 'runtime-source.json').read_text())
    record['files'] = {name: sha for name, sha in record['files'].items() if name not in OTHER_DEVICE_FILES[device]}
    for name, sha in record['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'invalid runtime-source.json path: {name}')
        source, target = runtime_dir / name, stage / 'runtime' / name
        # The copied record vouches for these files' official origin.
        if digest(source) != sha:
            raise ValueError(f'{source}: SHA-256 does not match runtime-source.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target, follow_symlinks=False)
    (stage / 'runtime/runtime-source.json').write_text(json.dumps(record, indent=2) + '\n')


def create(path, text, created):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        created.append(path)
        stream.write(text)


def main():
    root = Path(__file__).resolve().parents[1]
    prefix = Path(os.environ.get('YANLIU_PREFIX', Path.home() / '.local'))
    config_home = os.environ.get('XDG_CONFIG_HOME', '')
    config_dir = Path(config_home) if os.path.isabs(config_home) else Path.home() / '.config'
    config_file = config_dir / 'yanliu/config.json'
    component_path = os.environ.get('IBUS_COMPONENT_PATH', '/usr/share/ibus/component')
    # Both are written unquoted into the component <exec> and the environment.d file.
    if not re.fullmatch(r'/[a-zA-Z0-9_./-]*', str(prefix)):
        raise ValueError('YANLIU_PREFIX must be absolute and contain only letters, digits, _, ., / and -')
    if not re.fullmatch(r'[a-zA-Z0-9_./:-]+', component_path):
        raise ValueError('IBUS_COMPONENT_PATH must contain only letters, digits, _, ., /, : and -')
    device = os.environ.get('YANLIU_DEVICE', 'cpu')
    if device not in ('cpu', 'gpu', 'npu'):
        raise ValueError('YANLIU_DEVICE must be cpu, gpu or npu')
    accelerated = device != 'cpu'
    if accelerated and 'YANLIU_MODEL_DIR' not in os.environ:
        raise ValueError('gpu/npu need YANLIU_MODEL_DIR with source weights and matching IR')
    if accelerated != ('YANLIU_ACCELERATOR_RUNTIME_DIR' in os.environ):
        raise ValueError('gpu/npu need YANLIU_ACCELERATOR_RUNTIME_DIR; cpu does not use it')
    model_dir = Path(os.environ.get('YANLIU_MODEL_DIR', root / 'data/models'))
    source_model = model_dir / 'sentence-model.safetensors'
    existing_config = config_file.exists() or config_file.is_symlink()
    if existing_config:
        try:
            settings = json.loads(config_file.read_text())
            if not isinstance(settings, dict):
                raise ValueError('config must be an object')
            configured = settings.get('device', 'cpu')
        except ValueError as error:
            error.add_note(str(config_file))
            raise
        if configured != device:
            raise ValueError(f'{config_file} selects device {configured!r}; edit it or set YANLIU_DEVICE to match')
    destination = prefix / 'share/yanliu'
    launcher = prefix / 'bin/yanliu-ibus'
    component = prefix / 'share/ibus/component/yanliu.xml'
    environment = config_dir / 'environment.d/60-yanliu.conf'
    for path in (destination, launcher, component, environment):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite existing {path}; uninstall first.')

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.yanliu-install-', dir=destination.parent) as staging:
        stage = Path(staging) / 'yanliu'
        (stage / 'bin').mkdir(parents=True)
        (stage / 'models').mkdir()
        shutil.copy2(root / 'target/release/yanliu-ibus', stage / 'bin/yanliu-ibus')
        shutil.copytree(root / 'data/dictionaries', stage / 'dictionaries')
        shutil.copyfile(source_model, stage / 'models/sentence-model.safetensors')
        for name in ('LICENSE', 'NOTICE', 'README.md', 'provenance.json'):
            if (model_dir / name).is_file():
                shutil.copyfile(model_dir / name, stage / 'models' / name)
        shutil.copyfile(root / 'data/THIRD_PARTY.md', stage / 'THIRD_PARTY.md')
        shutil.copyfile(root / 'data/manifest.json', stage / 'source-manifest.json')
        (stage / 'licenses/chinese-ime-lm').mkdir(parents=True)
        shutil.copyfile(root / 'vendor/chinese-ime-lm/LICENSE', stage / 'licenses/chinese-ime-lm/LICENSE')
        (stage / 'licenses/chinese-ime-training').mkdir()
        shutil.copyfile(root / 'vendor/chinese-ime-training/NOTICE', stage / 'licenses/chinese-ime-training/NOTICE')
        if accelerated:
            copy_accelerator(model_dir,
                             Path(os.environ['YANLIU_ACCELERATOR_RUNTIME_DIR']), device, stage)
        files = {str(path.relative_to(stage)): digest(path) for path in sorted(stage.rglob('*')) if path.is_file()}
        (stage / 'installation.json').write_text(
            json.dumps({'format': 'yanliu-install-v1', 'model': {'source_path': str(source_model.resolve()),
                        'sha256': files['models/sentence-model.safetensors']}, 'files': files},
                       ensure_ascii=False, indent=2) + '\n')
        installed = []
        try:
            stage.rename(destination)
            installed.append(destination)
            launcher.parent.mkdir(parents=True, exist_ok=True)
            launcher.symlink_to(destination / 'bin/yanliu-ibus')
            installed.append(launcher)
            if not existing_config:
                settings = {'data_dir': str(destination)}
                if accelerated:
                    settings.update(device=device, runtime_dir=str(destination / 'runtime'))
                create(config_file, json.dumps(settings, ensure_ascii=False, indent=2) + '\n', installed)
            create(component, (root / 'packaging/yanliu.xml.in').read_text().replace('@PREFIX@', str(prefix))
                   .replace('@CONFIG@', escape(shlex.quote(str(config_file)))), installed)
            create(environment, f'IBUS_COMPONENT_PATH={prefix}/share/ibus/component:{component_path}\n', installed)
        except BaseException as error:
            for path in reversed(installed):
                try:
                    shutil.rmtree(path) if path.is_dir() else path.unlink()
                except OSError as cleanup:
                    error.add_note(f'rollback {path}: {cleanup}')
            raise
    print(f'Installed in {prefix}. Log out/in, then select 砚流 in GNOME Settings → Keyboard → Input Sources.')


if __name__ == '__main__':
    main()
