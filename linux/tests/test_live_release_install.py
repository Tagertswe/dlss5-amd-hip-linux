from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linux.tests.test_hip_install_safety import fixture
from dlssnr import deploy

class LiveReleaseInstall(unittest.TestCase):
    def setUp(self):
        # Keep the shared LD_PRELOAD bridge cache out of the real user home.
        self._xdg = tempfile.TemporaryDirectory()
        self._patcher = patch.dict(os.environ, {'XDG_DATA_HOME': self._xdg.name})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._xdg.cleanup()

    def test_live_pair_proxy_wrapper_and_uninstall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            for key in ('vkd3d', 'vkd3dcore'):
                artifacts[key] = root / key
                artifacts[key].write_bytes(key.encode())
            ini = exe.parent / 'ReShade.ini'
            ini.write_text('[GENERAL]\nTest=keep\n')
            original = ini.read_bytes()
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                result = deploy.install_hip(exe, weights, acknowledge_risk=True, replace_existing=True)
            self.assertTrue(result['valid'])
            self.assertEqual((exe.parent / 'dlss5-d3d12.dll').read_bytes(), b'vkd3d')
            self.assertEqual((exe.parent / 'd3d12core.dll').read_bytes(), b'vkd3dcore')
            self.assertIn('EnableProxyLibrary=1', ini.read_text())
            self.assertIn('ProxyLibrary=.\\dlss5-d3d12.dll', ini.read_text())
            wrapper = (exe.parent / '.dlssnr-linux/launch.sh').read_text()
            self.assertIn('version=b;', wrapper)
            self.assertNotIn('DRI_PRIME=', wrapper)
            self.assertFalse((exe.parent / 'DLSS5-AMD/continuous-every-frame.txt').exists())
            deploy.uninstall_game(exe, yes=True)
            self.assertEqual(ini.read_bytes(), original)
            self.assertFalse((exe.parent / 'dlss5-d3d12.dll').exists())

if __name__ == '__main__':
    unittest.main()
