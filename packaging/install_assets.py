#!/usr/bin/env python3
"""Install the built IBus service, dictionaries and selected ranker for this user; run by install-local.sh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from xml.sax.saxutils import escape
from xml.etree import ElementTree

# A GPU or NPU installation leaves out the other device's plugins.
OTHER_DEVICE_FILES = {'gpu': {'libopenvino_intel_npu_plugin.so', 'libopenvino_intel_npu_compiler.so'},
                      'npu': {'libopenvino_intel_gpu_plugin.so'}}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_ir(model_dir, source_dir=None):
    source_dir = source_dir or model_dir
    metadata_file = model_dir / 'sentence-model.openvino.json'
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    if metadata['format'] != 'yanliu-ranker-ir-v1':
        raise ValueError(f'{source_dir}/sentence-model.openvino.json: unsupported ranker metadata')
    for suffix, field in (('safetensors', 'source_sha256'), ('xml', 'xml_sha256'), ('bin', 'bin_sha256')):
        if digest(model_dir / f'sentence-model.{suffix}') != metadata[field]:
            raise ValueError(f'{source_dir}/sentence-model.{suffix}: SHA-256 does not match ranker metadata')


def copy_accelerator(model_dir, runtime_dir, device, stage):
    for suffix in ('xml', 'bin', 'openvino.json'):
        shutil.copyfile(model_dir / f'sentence-model.{suffix}', stage / f'models/sentence-model.{suffix}')
    verify_ir(stage / 'models', model_dir)
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


def installation(directory):
    manifest = directory / 'installation.json'
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f'{manifest}: expected a regular installation manifest')
    record = json.loads(manifest.read_text())
    if not isinstance(record, dict) or record.get('format') != 'yanliu-install-v1':
        raise ValueError(f'{directory}: unsupported installation manifest')
    if set(record) == {'format', 'model', 'files'}:
        model = record['model']
        if not isinstance(model, dict) or set(model) != {'source_path', 'sha256'} or not isinstance(model['source_path'], str):
            raise ValueError(f'{directory}: invalid selected model record')
    elif set(record) == {'format', 'selected_model', 'model_provenance', 'files'}:
        if not isinstance(record['selected_model'], str) or record['model_provenance'] != 'models/provenance.json':
            raise ValueError(f'{directory}: invalid selected model record')
    else:
        raise ValueError(f'{directory}: unsupported installation manifest fields')
    files = record['files']
    if not isinstance(files, dict) or not {'bin/yanliu-ibus', 'models/sentence-model.safetensors'} <= files.keys():
        raise ValueError(f'{directory}: installation lacks the managed binary or model')
    for name, expected in files.items():
        if not isinstance(name, str):
            raise ValueError(f'{directory}: invalid installation filename {name!r}')
        path = Path(name)
        if (not name or path.is_absolute() or '..' in path.parts or str(path) != name
                or name == 'installation.json' or not isinstance(expected, str)
                or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError(f'{directory}: invalid installation entry {name!r}')
        target = directory / path
        if any(parent.is_symlink() for parent in target.parents if parent != directory and directory in parent.parents):
            raise ValueError(f'{target}: managed directory must not be a symlink')
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f'{target}: managed file escapes installation')
        if not target.is_file() or digest(target) != expected:
            raise ValueError(f'{target}: installed SHA-256 does not match installation.json')
    if 'model' in record and record['model']['sha256'] != files['models/sentence-model.safetensors']:
        raise ValueError(f'{directory}: selected model SHA-256 does not match files')
    if 'model_provenance' in record and record['model_provenance'] not in files:
        raise ValueError(f'{directory}: selected model provenance is not managed')
    return files


def unmanaged(directory, files):
    managed = {Path(name) for name in files} | {Path('installation.json')}
    parents = {parent for path in managed for parent in path.parents}
    result = []

    def visit(path):
        relative = path.relative_to(directory)
        if relative in managed:
            return
        if relative in parents and path.is_dir() and not path.is_symlink():
            for child in sorted(path.iterdir()):
                visit(child)
        else:
            result.append(relative)

    for path in sorted(directory.iterdir()):
        visit(path)
    return result


def check_unmanaged_conflicts(stage, extra):
    for name in extra:
        target = stage / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'{target}: new assets conflict with an unmanaged file')
        for parent in target.parents:
            if parent == stage:
                break
            if parent.exists() and (not parent.is_dir() or parent.is_symlink()):
                raise FileExistsError(f'{parent}: new assets conflict with an unmanaged directory')


def managed_layout(destination, launcher, component, environment, config_file, settings, device):
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f'{destination}: expected a managed installation directory')
    target = destination / 'bin/yanliu-ibus'
    if launcher.is_symlink():
        managed_launcher = launcher.readlink() == target
    else:
        legacy = f'#!/bin/sh\nset -eu\nexec "{target}" "$@"\n'.encode()
        managed_launcher = (launcher.is_file() and launcher.stat().st_mode & 0o7777 == 0o755
                            and launcher.read_bytes() == legacy)
    if not managed_launcher:
        raise ValueError(f'{launcher}: does not point to the managed service')
    xml = ElementTree.parse(component).getroot()
    if (xml.findtext('name') != 'org.freedesktop.IBus.Yanliu'
            or shlex.split(xml.findtext('exec', '')) != [str(launcher), '--ibus', '--config', str(config_file)]):
        raise ValueError(f'{component}: does not use the managed service and config')
    if not environment.is_file() or environment.is_symlink():
        raise ValueError(f'{environment}: expected the installed environment file')
    if not environment.read_text().startswith(f'IBUS_COMPONENT_PATH={component.parent}:'):
        raise ValueError(f'{environment}: does not include the managed component directory')
    if not settings:
        raise ValueError(f'{config_file}: upgrade needs an explicit managed data_dir')
    allowed = {'data_dir', 'model', 'device', 'runtime_dir', 'learning_enabled', 'contextual_learning', 'user_dictionary'}
    if settings.keys() - allowed:
        raise ValueError(f'{config_file}: unknown config fields: {sorted(settings.keys() - allowed)}')
    if (device == 'cpu') == ('runtime_dir' in settings):
        raise ValueError(f'{config_file}: runtime_dir is required for gpu/npu and must be absent for cpu')
    for name, expected in (('data_dir', destination), ('model', destination / 'models/sentence-model.safetensors'),
                           ('runtime_dir', destination / 'runtime')):
        if name not in settings and name != 'data_dir':
            continue
        value = settings.get(name)
        if not isinstance(value, str) or (config_file.parent / value).resolve() != expected.resolve():
            raise ValueError(f'{config_file}: {name} does not point to the managed installation')
    return installation(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--upgrade', action='store_true', help='replace a verified installation, retaining the previous version')
    action.add_argument('--rollback', action='store_true', help='swap the verified current and previous asset versions')
    args = parser.parse_args()
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
    if accelerated and 'YANLIU_MODEL_DIR' not in os.environ and not args.rollback:
        raise ValueError('gpu/npu need YANLIU_MODEL_DIR with source weights and matching IR')
    if not args.rollback and accelerated != ('YANLIU_ACCELERATOR_RUNTIME_DIR' in os.environ):
        raise ValueError('gpu/npu need YANLIU_ACCELERATOR_RUNTIME_DIR; cpu does not use it')
    model_dir = Path(os.environ.get('YANLIU_MODEL_DIR', root / 'data/models'))
    source_model = model_dir / 'sentence-model.safetensors'
    existing_config = config_file.exists() or config_file.is_symlink()
    settings = None
    if existing_config:
        try:
            settings = json.loads(config_file.read_text())
            if not isinstance(settings, dict):
                raise ValueError('config must be an object')
            configured = settings.get('device', 'cpu')
        except ValueError as error:
            error.add_note(str(config_file))
            raise
        if configured not in ('cpu', 'gpu', 'npu'):
            raise ValueError(f'{config_file}: device must be cpu, gpu or npu')
        if args.rollback:
            device = configured
        elif configured != device:
            raise ValueError(f'{config_file} selects device {configured!r}; edit it or set YANLIU_DEVICE to match')
    destination = prefix / 'share/yanliu'
    launcher = prefix / 'bin/yanliu-ibus'
    component = prefix / 'share/ibus/component/yanliu.xml'
    environment = config_dir / 'environment.d/60-yanliu.conf'
    previous = destination.parent / '.yanliu-previous'
    extra = []
    if args.upgrade or args.rollback:
        files = managed_layout(destination, launcher, component, environment, config_file, settings, device)
        extra = unmanaged(destination, files)
        if previous.exists() or previous.is_symlink():
            if previous.is_symlink() or not previous.is_dir():
                raise ValueError(f'{previous}: expected a managed previous installation directory')
            previous_files = installation(previous)
            if args.rollback and device != 'cpu':
                required = {f'models/sentence-model.{suffix}' for suffix in ('xml', 'bin', 'openvino.json')}
                required |= {'runtime/runtime-source.json', f'runtime/libopenvino_intel_{device}_plugin.so'}
                if not required <= previous_files.keys():
                    raise ValueError(f'{previous}: previous assets do not support configured device {device}')
                verify_ir(previous / 'models')
            if unmanaged(previous, previous_files):
                raise ValueError(f'{previous}: previous installation contains unmanaged files; preserve them before replacing it')
        elif args.rollback:
            raise FileNotFoundError(f'{previous}: no previous installation to restore')
    else:
        for path in (destination, launcher, component, environment):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f'Refusing to overwrite existing {path}; use --upgrade for a managed installation.')

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.yanliu-install-', dir=destination.parent))
    cleanup_staging = True
    try:
        stage = staging / 'yanliu'
        if not args.rollback:
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
        if args.upgrade or args.rollback:
            check_unmanaged_conflicts(previous if args.rollback else stage, extra)
            saved_previous = None
            saved_current = False
            replacement_active = False
            moved = []
            try:
                if previous.exists():
                    target = stage if args.rollback else staging / 'older'
                    previous.rename(target)
                    saved_previous = target
                destination.rename(previous)
                saved_current = True
                stage.rename(destination)
                replacement_active = True
                # Unknown contents move to the current version without being read.
                for name in extra:
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (previous / name).rename(target)
                    moved.append(name)
            except BaseException as error:
                try:
                    for name in reversed(moved):
                        (destination / name).rename(previous / name)
                    if replacement_active:
                        destination.rename(stage)
                    if saved_current:
                        previous.rename(destination)
                    if saved_previous is not None:
                        saved_previous.rename(previous)
                except BaseException as cleanup:
                    cleanup_staging = False
                    error.add_note(f'restoring installation failed: {cleanup}; recovery assets preserved in {staging}')
                raise
            verb = 'Restored previous' if args.rollback else 'Upgraded'
            print(f'{verb} assets in {destination}; replaced assets are in {previous}. '
                  'Reload the Yanliu IBus service before typing. Configuration and learning were not changed.')
            return
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
    finally:
        if cleanup_staging:
            shutil.rmtree(staging)
    print(f'Installed in {prefix}. Log out/in, then select 砚流 in GNOME Settings → Keyboard → Input Sources.')


if __name__ == '__main__':
    main()
