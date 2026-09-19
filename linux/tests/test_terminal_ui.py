#!/usr/bin/env python3
import contextlib
import fcntl
import io
import os
import pty
import select
import struct
import sys
import tempfile
import termios
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlssnr import VERSION, cli, terminal


def _write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _pe64() -> bytes:
    """Minimal valid PE x64 executable (machine 0x8664, not a DLL)."""
    buf = bytearray(0x110)
    buf[0:2] = b'MZ'
    struct.pack_into('<I', buf, 0x3c, 0x40)
    buf[0x40:0x44] = b'PE\x00\x00'
    struct.pack_into('<HHIIIHH', buf, 0x44, 0x8664, 1, 0, 0, 0, 112, 0x0100)
    opt = 0x58
    struct.pack_into('<H', buf, opt, 0x20b)
    struct.pack_into('<Q', buf, opt + 24, 0x140000000)
    struct.pack_into('<I', buf, opt + 60, 240)
    struct.pack_into('<I', buf, opt + 108, 0)
    sec = opt + 112
    buf[sec:sec + 8] = b'text\x00\x00\x00'
    struct.pack_into('<IIIIIIHHI', buf, sec + 8, 0, 0x2000, 0, 240, 0, 0, 0, 0, 0x40000000)
    return bytes(buf)


class TerminalUiTests(unittest.TestCase):
    def test_non_tty_falls_back(self):
        old_out = sys.stdout
        try:
            sys.stdout = io.StringIO()
            self.assertFalse(terminal.Tui().start())
        finally:
            sys.stdout = old_out

    def test_narrow_terminal_falls_back(self):
        master, slave = pty.openpty()
        old_in, old_out = sys.stdin, sys.stdout
        old_cols, old_lines = os.environ.get('COLUMNS'), os.environ.get('LINES')
        try:
            sys.stdin = os.fdopen(slave, 'r', encoding='utf-8', closefd=False)
            sys.stdout = os.fdopen(slave, 'w', encoding='utf-8', closefd=False)
            os.environ['COLUMNS'], os.environ['LINES'] = '80', '24'
            self.assertFalse(terminal.Tui().start())
        finally:
            sys.stdin, sys.stdout = old_in, old_out
            os.close(master)
            os.close(slave)
            if old_cols is None:
                os.environ.pop('COLUMNS', None)
            else:
                os.environ['COLUMNS'] = old_cols
            if old_lines is None:
                os.environ.pop('LINES', None)
            else:
                os.environ['LINES'] = old_lines

    def test_wide_terminal_keeps_art_and_viewport(self):
        master, slave = pty.openpty()
        old_in, old_out = sys.stdin, sys.stdout
        old_cols, old_lines = os.environ.get('COLUMNS'), os.environ.get('LINES')
        try:
            sys.stdin = os.fdopen(slave, 'r', encoding='utf-8', closefd=False)
            sys.stdout = os.fdopen(slave, 'w', encoding='utf-8', closefd=False)
            os.environ['COLUMNS'], os.environ['LINES'] = '140', '40'
            tui = terminal.Tui()
            self.assertTrue(tui.start())
            self.assertEqual(tui.columns, 140)
            sys.stdout.write('line one\n')
            sys.stdout.write('line two\n')
            sys.stdout.write('progress 10\rprogress 90\n')
            os.write(master, b'y\n')
            self.assertEqual(input('choice [y/N] '), 'y')
            sys.stdout.write('final\n')
            tui.stop()
            self.assertNotIsInstance(sys.stdout, terminal._Shell)
            data = b''
            while True:
                ready, _, _ = select.select([master], [], [], 0.3)
                if not ready:
                    break
                chunk = os.read(master, 65536)
                if not chunk:
                    break
                data += chunk
            text = data.decode('utf-8', 'replace')
            self.assertIn('line one', text)
            self.assertIn('line two', text)
            self.assertIn('choice [y/N] y', text)
            self.assertIn('dlss5-amd-hip ' + VERSION, text)
            self.assertIn(terminal.ART.splitlines()[3].strip(), text)
        finally:
            sys.stdin, sys.stdout = old_in, old_out
            os.close(master)
            os.close(slave)
            if old_cols is None:
                os.environ.pop('COLUMNS', None)
            else:
                os.environ['COLUMNS'] = old_cols
            if old_lines is None:
                os.environ.pop('LINES', None)
            else:
                os.environ['LINES'] = old_lines


class NonSteamTests(unittest.TestCase):
    def test_interactive_manual_path_without_steam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / 'mygame' / 'Game.exe'
            _write(exe, _pe64())
            (root / 'empty').mkdir()
            args = cli.parser().parse_args(['install'])
            with patch.object(cli.games, 'discover_games', return_value=[]), \
                 patch.object(cli, 'PACKAGE_ROOT', root / 'empty'), \
                 patch('builtins.input', side_effect=[str(root / 'mygame')]):
                selected = cli.resolve_exe(args, interactive=True)
            self.assertEqual(selected, exe)

    def test_interactive_sentinel_selects_manual_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / 'mygame' / 'Game.exe'
            _write(exe, _pe64())
            args = cli.parser().parse_args(['install'])
            entries = [{'appid': '1', 'name': 'Fake Steam Game', 'path': root / 'steamgame'}]
            with patch.object(cli.games, 'discover_games', return_value=entries), \
                 patch('builtins.input', side_effect=[str(len(entries) + 1), str(root / 'mygame')]):
                selected = cli.resolve_exe(args, interactive=True)
            self.assertEqual(selected, exe)

    def test_game_dir_flag_never_touches_steam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / 'mygame' / 'Game.exe'
            _write(exe, _pe64())
            args = cli.parser().parse_args(['install', '--game-dir', str(root / 'mygame')])
            with patch.object(cli.games, 'discover_games', side_effect=AssertionError('Steam lookup')):
                selected = cli.resolve_exe(args, interactive=False)
            self.assertEqual(selected, exe)


class BannerTests(unittest.TestCase):
    def test_status_prints_version_banner(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / 'Game.exe'
            _write(exe, _pe64())
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = cli.main(['status', '--exe', str(exe)])
            self.assertEqual(code, 0)
            self.assertIn(f'dlss5-amd-hip {VERSION} —', buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
