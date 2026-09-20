"""Real HIP enumeration, no model or game required."""
import ctypes
from ctypes.util import find_library
import os
from pathlib import Path
import sys
import unittest

LIBRARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv)>1 else Path('hip/libdlss5_hip.so').resolve()


def _amdhip64():
    names = ('libamdhip64.so.7', 'libamdhip64.so')
    roots = []
    for key in ('HIP_LIB', 'ROCM_PATH', 'HIP_PATH'):
        value = os.environ.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_file():
            return path
        roots.append(path)
    if Path('/opt').is_dir():
        roots.extend(sorted(Path('/opt').glob('rocm*'), reverse=True))
    for root in roots:
        for sub in ('lib', 'lib64'):
            for name in names:
                candidate = root / sub / name
                if candidate.is_file():
                    return candidate
    return find_library('amdhip64')


class DeviceApiTests(unittest.TestCase):
    def test_exact_device_match_and_reject_missing(self):
        lib=ctypes.CDLL(str(LIBRARY))
        self.assertTrue(hasattr(lib,'dlss5_find_device'),'missing graphics-to-HIP adapter matching API')
        lookup=lib.dlss5_find_device
        lookup.argtypes=[ctypes.c_char_p];lookup.restype=ctypes.c_int
        lib.dlss5_last_error.restype=ctypes.c_char_p
        runtime=_amdhip64()
        if not runtime:
            self.skipTest('libamdhip64 not found; set ROCM_PATH or HIP_LIB')
        hip=ctypes.CDLL(str(runtime))
        count=ctypes.c_int();self.assertEqual(hip.hipGetDeviceCount(ctypes.byref(count)),0)
        names=[]
        for i in range(count.value):
            name=ctypes.create_string_buffer(256)
            self.assertEqual(hip.hipDeviceGetName(name,256,i),0)
            names.append(name.value)
        for i,name in enumerate(names):
            self.assertEqual(lookup(name),i if names.count(name)==1 else -1)
        self.assertEqual(lookup(b'nonexistent graphics adapter'),-1)
        self.assertIn(b'not found',lib.dlss5_last_error())
        self.assertEqual(lookup(None),-1)

if __name__=='__main__':unittest.main()
