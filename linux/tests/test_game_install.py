#!/usr/bin/env python3
import hashlib
import os
import re
import shlex
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlssnr import addon, deploy, package


def _write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _fake_package(root: Path, *, magpie=False):
    loader = 'dxgi.dll' if magpie else 'd3d12.dll'
    _write(root / 'dlss5-amd.addon64', b'MZ' + b'\0' * (64 * 1024))
    _write(root / loader, b'ReShade')
    _write(root / 'DLSS5-AMD' / 'native-game-flags.txt', b'DLSS5_TILED_WEIGHTS=1\n')
    _write(root / 'DLSS5-AMD' / 'native-game-tiled-assets' / 'block0.f16', b'\0\0')
    _write(root / 'DLSS5-AMD' / 'native-game-tiled-assets' / 'wave.cso', b'CSO')
    _write(root / 'DLSS5-D3D12-721' / 'D3D12Core.dll', b'core')
    return root


class PackageTests(unittest.TestCase):
    def test_validate_game_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_package(Path(tmp) / 'pkg')
            info = package.validate(root)
            self.assertEqual(info['mode'], 'game')
            self.assertTrue(info['agility_sdk'])
            self.assertEqual(info['weights'], 1)
            self.assertEqual(info['loader'], 'd3d12.dll')

    def test_rejects_hip_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / 'amdhip64_7.dll', b'nope')
            with self.assertRaises(RuntimeError):
                package.validate(root)

    def test_magpie_needs_dxgi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_package(Path(tmp) / 'pkg')
            with self.assertRaises(RuntimeError):
                package.validate(root, magpie=True)
            _fake_package(Path(tmp) / 'mag', magpie=True)
            info = package.validate(Path(tmp) / 'mag', magpie=True)
            self.assertEqual(info['loader'], 'dxgi.dll')


class DeployTests(unittest.TestCase):
    def test_wrapper_overrides_d3d12_not_hip(self):
        exe = Path('/tmp/game/Game.exe')
        text = deploy.wrapper_bytes(exe).decode()
        self.assertIn('d3d12=n,b', text)
        self.assertIn('d3d12core=n,b', text)
        self.assertNotIn('amdhip64', text)
        self.assertNotIn('LD_PRELOAD', text)
        magpie = deploy.wrapper_bytes(exe, magpie=True).decode()
        self.assertIn('dxgi=n,b', magpie)

    def test_install_and_uninstall_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pkg = _fake_package(tmp / 'pkg')
            game = tmp / 'game'
            exe = game / 'Game.exe'
            _write(exe, b'MZ')
            existing = game / 'd3d12.dll'
            _write(existing, b'original-reshade')
            result = deploy.install_package(
                exe, pkg, acknowledge_risk=True, replace_existing=True)
            self.assertTrue(result['valid'])
            self.assertTrue((game / 'dlss5-amd.addon64').is_file())
            self.assertTrue((game / 'DLSS5-AMD' / 'native-game-flags.txt').is_file())
            self.assertTrue((game / '.dlssnr-linux' / 'launch.sh').is_file())
            self.assertEqual((game / 'd3d12.dll').read_bytes(), b'ReShade')
            removed = deploy.uninstall_game(exe, yes=True)
            self.assertTrue(removed['removed'])
            self.assertEqual((game / 'd3d12.dll').read_bytes(), b'original-reshade')
            self.assertFalse((game / 'dlss5-amd.addon64').exists())
            self.assertFalse((game / '.dlssnr-linux').exists())


FAKE_FOREIGN_UNINSTALLER = b'''#!/usr/bin/env python3
import json, shutil, sys
from pathlib import Path
args = sys.argv[1:]
if args[:1] != ['uninstall'] or '--exe' not in args or '--yes' not in args:
    raise SystemExit(2)
exe = Path(args[args.index('--exe') + 1])
game = exe.parent
store = game / '.dlssnr-linux'
data = json.loads((store / 'manifest.json').read_text())
if data.get('schema') != 1 or data.get('exe') != str(exe):
    raise SystemExit('refusing foreign manifest')
for name in data['files']:
    target = game / name
    if target.exists():
        target.unlink()
shutil.rmtree(store)
print(json.dumps({'installed': False, 'valid': False, 'removed': True}))
'''


def _foreign_item(payload: bytes, *, preserve: bool = False) -> dict:
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'mode': 0o644,
            'original': None, 'original_mode': None, 'preserve': preserve, 'touched': True}


def _install_foreign(game: Path, exe: Path, *, drift_ini: bool = False, staging: bool = True):
    payloads = {'d3d12.dll': b'foreign-d3d12', 'd3d12core.dll': b'foreign-core',
                'dlssnr_on_amd.ini': b'[PROXY]\n', 'dlssnr_on_amd_weights.bin': b'w'}
    files = {name: _foreign_item(data) for name, data in payloads.items()}
    store = game / deploy.STORE
    (store / 'backups').mkdir(parents=True, exist_ok=True)
    journal = {'schema': 1, 'exe': str(exe), 'state': 'installed', 'files': files,
               'request': 'f09c0a6347c91137a95635e2c5fc5991e8227063eca630f03b777b2677a6433a',
               'cache': str(game / 'cache'), 'undo': {}}
    _write(store / 'manifest.json', json.dumps(journal, indent=2).encode())
    on_disk = dict(payloads)
    if drift_ini:
        on_disk['dlssnr_on_amd.ini'] = b'[PROXY]\nRewrittenAtRuntime=1\n'
    for name, data in on_disk.items():
        _write(game / name, data)
    if staging:
        staging_dir = game / 'dlssnr-linux-portable'
        (staging_dir / 'dlssnr').mkdir(parents=True)
        _write(staging_dir / 'PROVENANCE.json', b'{"schema": 1}')
        _write(staging_dir / 'installer.py', FAKE_FOREIGN_UNINSTALLER)


class ForeignDeploymentTests(unittest.TestCase):
    def test_foreign_journal_is_detected_and_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = _fake_package(root / 'pkg')
            game = root / 'game'
            exe = game / 'Game.exe'
            _write(exe, b'MZ')
            _install_foreign(game, exe)
            self.assertIsNotNone(deploy.foreign_deployment(exe))
            status = deploy.status_game(exe)
            self.assertFalse(status['installed'])
            self.assertTrue(status.get('foreign'))
            self.assertTrue(any('--replace-foreign' in note for note in status['notes']))
            with self.assertRaisesRegex(RuntimeError, r'--replace-foreign'):
                deploy.install_package(exe, pkg, acknowledge_risk=True, replace_existing=True)
            self.assertTrue((game / 'd3d12.dll').exists())

    def test_replace_foreign_removes_then_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = _fake_package(root / 'pkg')
            game = root / 'game'
            exe = game / 'Game.exe'
            _write(exe, b'MZ')
            _install_foreign(game, exe, drift_ini=True)
            removed = deploy.remove_foreign_deployment(exe)
            self.assertEqual(len(removed['files']), 4)
            self.assertFalse((game / deploy.STORE).exists())
            self.assertFalse((game / 'd3d12.dll').exists())
            result = deploy.install_package(exe, pkg, acknowledge_risk=True)
            self.assertTrue(result['valid'])
            journal = json.loads((game / deploy.STORE / 'manifest.json').read_text())
            self.assertEqual(journal['kind'], 'dlss5')

    def test_replace_foreign_requires_staged_uninstaller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / 'game'
            exe = game / 'Game.exe'
            _write(exe, b'MZ')
            _install_foreign(game, exe, staging=False)
            with self.assertRaisesRegex(RuntimeError, 'installer'):
                deploy.remove_foreign_deployment(exe)
            self.assertTrue((game / deploy.STORE).exists())

    def test_unknown_journal_is_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = _fake_package(root / 'pkg')
            game = root / 'game'
            exe = game / 'Game.exe'
            _write(exe, b'MZ')
            store = game / deploy.STORE
            store.mkdir()
            other = {'schema': 9, 'kind': 'otherproduct', 'exe': str(exe),
                     'state': 'installed', 'files': {}}
            _write(store / 'manifest.json', json.dumps(other).encode())
            self.assertIsNone(deploy.foreign_deployment(exe))
            with self.assertRaisesRegex(RuntimeError, 'refusing mutation'):
                deploy.install_package(exe, pkg, acknowledge_risk=True, replace_existing=True)


class BackendIntentTests(unittest.TestCase):
    def test_d3d_default_never_looks_up_hip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = _fake_package(root / 'pkg')
            exe = root / 'game' / 'Game.exe'
            _write(exe, b'MZ')
            with patch.object(deploy, 'ensure_hip_artifacts', side_effect=AssertionError('HIP lookup in D3D')):
                result = deploy.install_package(exe, pkg, acknowledge_risk=True)
            self.assertFalse(result['hip'])
            self.assertFalse(deploy.status_game(exe)['hip'])
            journal = json.loads((exe.parent / deploy.STORE / 'manifest.json').read_text())
            self.assertFalse(journal['hip'])
            self.assertFalse((exe.parent / 'dlss5_hip.dll').exists())
            self.assertFalse((exe.parent / deploy.STORE / 'lib').exists())
            self.assertNotIn('LD_PRELOAD', (exe.parent / deploy.STORE / 'launch.sh').read_text())
            self.assertTrue(any('SM 6.10' in note for note in result['notes']))
            self.assertFalse(any('HIP path:' in note for note in result['notes']))

    def _hip_install(self, root, game):
        weights = root / 'weights'
        _write(weights / 'block0.f16', b'\0\0')
        artifacts = {key: root / 'artifacts' / key for key in ('so', 'dll', 'addon', 'reshade', 'flags')}
        for key, path in artifacts.items():
            _write(path, ('fixture-' + key).encode() if key != 'flags' else b'DLSS5_TILED_WEIGHTS=1\n')
        exe = game / 'Game.exe'
        _write(exe, b'MZ')
        xdg = root / 'xdg'
        with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts), \
             patch.dict(os.environ, {'XDG_DATA_HOME': str(xdg)}):
            result = deploy.install_package(exe, weights, acknowledge_risk=True, hip=True)
        return exe, xdg, result

    def test_explicit_hip_stays_hip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, xdg, result = self._hip_install(root, root / 'game')
            self.assertTrue(result['hip'])
            self.assertTrue(deploy.status_game(exe)['hip'])
            self.assertEqual((exe.parent / 'dlss5_hip.dll').read_bytes(), b'fixture-dll')
            self.assertEqual((exe.parent / deploy.STORE / 'lib/libdlss5_hip.so').read_bytes(), b'fixture-so')
            wrapper = (exe.parent / deploy.STORE / 'launch.sh').read_text()
            self.assertIn('export LD_PRELOAD=', wrapper)
            self.assertIn('export DLSS5_HIP=1', wrapper)
            preload = self._preload_target(wrapper)
            self.assertTrue(preload.startswith(str(xdg)), preload)
            self.assertEqual(Path(preload).read_bytes(), b'fixture-so')
            journal = json.loads((exe.parent / deploy.STORE / 'manifest.json').read_text())
            self.assertEqual(journal['bridge_cache'], preload)
            self.assertEqual(deploy.status_game(exe)['bridge_cache'], preload)
            # A staged diagnostic must not arm known wait-based live rendering.
            for name in ('continuous-every-frame.txt', 'continuous-reset-preview.txt',
                         'temporal-history.txt', 'neural-frame-request.txt'):
                self.assertFalse((exe.parent / 'DLSS5-AMD' / name).exists(), name)
            self.assertTrue(any('unverified' in note.lower() for note in result['notes']))
            self.assertFalse((exe.parent / 'DLSS5-D3D12-721').exists())

    @staticmethod
    def _preload_target(wrapper):
        # The wrapper exports LD_PRELOAD="$so..."; the target is the so= line.
        match = re.search(r'^so=(.+)$', wrapper, re.M)
        if not match:
            raise AssertionError('no so= line in wrapper')
        return shlex.split(match.group(1))[0]

    def test_hip_preload_target_survives_whitespace_game_path(self):
        # 'SILENT HILL f': the game directory contains a space, which the ELF
        # loader would split out of LD_PRELOAD, so the preload target must be
        # the whitespace-free cache copy, never the in-game staged library.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / 'SILENT HILL f' / 'SHf' / 'Binaries' / 'Win64'
            exe, xdg, result = self._hip_install(root, game)
            self.assertTrue(result['valid'])
            wrapper = (exe.parent / deploy.STORE / 'launch.sh').read_text()
            preload = self._preload_target(wrapper)
            self.assertNotIn(' ', preload)
            self.assertTrue(preload.startswith(str(xdg)), preload)
            self.assertEqual(Path(preload).read_bytes(), b'fixture-so')
            # The in-game staged copy stays journal-managed but is not preloaded.
            self.assertEqual((game / deploy.STORE / 'lib/libdlss5_hip.so').read_bytes(), b'fixture-so')
            self.assertNotIn(str(game), preload)

    def test_hip_bridge_cache_refuses_whitespace_data_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / 'weights'
            _write(weights / 'block0.f16', b'\0\0')
            artifacts = {key: root / 'artifacts' / key for key in ('so', 'dll', 'addon', 'reshade', 'flags')}
            for key, path in artifacts.items():
                _write(path, ('fixture-' + key).encode() if key != 'flags' else b'DLSS5_TILED_WEIGHTS=1\n')
            game = root / 'game'
            exe = game / 'Game.exe'
            _write(exe, b'MZ')
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts), \
                 patch.dict(os.environ, {'XDG_DATA_HOME': str(root / 'bad data home')}), \
                 self.assertRaisesRegex(RuntimeError, 'whitespace'):
                deploy.install_package(exe, weights, acknowledge_risk=True, hip=True)

    def test_explicit_d3d_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = _fake_package(root / 'pkg')
            exe = root / 'game' / 'Game.exe'
            _write(exe, b'MZ')
            before = sorted(p.relative_to(root) for p in root.rglob('*'))
            result = deploy.install_package(exe, pkg, acknowledge_risk=True, hip=False, dry_run=True)
            self.assertFalse(result['hip'])
            self.assertEqual(before, sorted(p.relative_to(root) for p in root.rglob('*')))


class AddonTests(unittest.TestCase):
    def test_mingw_error_is_explicit(self):
        try:
            addon.mingw_cxx()
        except RuntimeError as exc:
            self.assertIn('mingw-w64', str(exc))
        else:
            self.assertTrue(addon.mingw_cxx())


if __name__ == '__main__':
    unittest.main()
