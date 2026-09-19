"""Fixed right-panel ASCII art for the interactive installer session.

Output stays in a no-scroll left viewport while the art on the right never
moves. Falls back to plain output for non-TTY, dumb or narrow terminals.
"""
from __future__ import annotations

import builtins
import os
import shutil
import sys

from . import VERSION, TAGLINE

MIN_LEFT = 44
GAP = 2

ART = '''\
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡤⠲⠒⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡌⠣⡇⠀⠀⢱⠱⠀⠀⠀⠀⠀⠀⠀⢀⣤⣾⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣟⠈⡇⠀⠀⢺⡰⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⠸⡀⠘⢿⠔⠆⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣧⣤⣤⣤⣤⣤⡄⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⣱⣾⠁⢽⢿⣿⣗⡖⢢⠄⣻⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡔⣺⢫⠋⠉⠀⢿⣧⠹⡻⡌⢙⠾⣝⣻⣿⣿⣿⣿⡛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢲⣶⣶⣶⣾⣿⣿⣦⡀⠀⠀⡰⠁⢰⠇⣸⠀⠀⠀⢸⡟⣆⢧⡟⡵⡤⠚⠳⡈⢫⠙⡻⣡⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠙⠿⣿⣿⣿⣿⣿⣿⣆⠜⣰⢄⢸⣀⢿⡤⣧⣀⢼⢹⡽⡞⣫⢝⡌⠀⠀⠘⡄⢣⠀⢡⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠈⢙⣿⣿⣿⣿⡟⢺⠣⢴⢻⡯⣿⣽⠙⠒⣽⠏⣇⣧⡧⢧⢿⠤⠀⠆⢹⡈⡆⠈⣼⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣴⣿⣿⣿⣿⠏⢠⠋⠀⠘⣸⠀⣿⠀⠀⠜⢸⣿⢹⡟⡷⠊⣾⣧⡀⢸⠀⣧⣣⠀⣷⣃⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⣾⣿⣿⣿⣿⡿⢠⣿⢰⠀⡆⡿⠀⣿⣧⠀⠀⣧⣇⣼⢾⣿⣿⣿⢿⡧⡸⡰⢻⡽⡊⣈⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠴⠿⠿⠛⢛⡿⠛⣇⣿⢿⠘⡀⢁⠅⡖⡗⢻⠀⢠⢿⢩⠏⢾⣿⡿⣿⣸⢡⠐⡤⢺⣡⠉⢰⢳⡆⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⡿⠀⣿⣷⠘⣧⣷⠃⣈⣿⣯⣽⡄⣌⠂⠀⠀⠀⠹⠓⠈⢸⢸⠀⠀⠀⣽⡀⠈⢸⣥⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠘⣇⠀⡏⡏⡾⡹⣻⡆⣿⠹⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⢘⣿⢸⢀⢠⢿⡇⠀⠘⣟⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⡄⣿⠀⣳⣷⢷⠉⡜⣿⣿⡁⠹⠟⠊⠀⠀⠀⠀⠀⠀⠀⠀⣾⢺⢸⡘⣸⡾⡇⠀⡄⡟⡇⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⡗⣿⠀⢿⣿⣿⣣⠸⡘⣿⢿⠦⡀⠀⠀⠀⠀⢾⠋⡱⢀⠴⡣⡏⡏⣇⠻⣇⣿⡆⠃⡇⣷⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣧⠛⠂⣞⣿⠧⠑⠵⡑⣝⡿⣧⣑⠤⣀⡀⠀⠀⠁⠀⣠⠊⢧⣿⣿⠎⠀⣿⣻⠘⢠⢀⢸⡄⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⡰⠁⢠⣞⣻⡽⠀⣔⠦⢬⠻⣿⣦⡉⠁⠀⠀⡩⡏⢀⠞⠁⢀⣼⡿⡏⠀⠀⣿⣿⣧⢸⢸⠂⣇⠀⠀⠀⠀⠀⠀
⠀⠀⠀⡐⠀⢀⠉⠄⠊⢱⠘⢦⣄⠍⠀⢨⢯⢍⡓⠶⣖⡏⡷⡅⠀⡜⢣⢫⢺⣧⡀⠀⡇⢹⣿⣯⡟⠀⢹⠀⠀⠀⠀⠀⠀
⠀⠀⣰⠃⠀⠁⠠⠐⠚⢉⡆⠀⠀⠀⡠⢃⠉⠇⠉⡲⣀⣽⣷⣿⣼⣶⣿⡅⢸⢰⡹⠉⠓⠤⣻⣿⢳⡖⠚⡇⠀⠀⠀⠀⠀
⠀⠀⡏⠀⠀⠀⠀⠀⣰⠟⠀⠀⠀⡔⢀⢼⡠⠀⠼⠚⠀⠛⣿⣿⣿⠟⠑⠡⣈⠆⠀⠀⠀⠀⣀⣍⡺⡇⠀⢻⠀⠀⠀⠀⠀
⢀⠴⠀⠀⠀⠀⠀⢠⡇⠀⠀⢀⡜⠀⣜⠝⠀⠀⠀⠀⠀⣸⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⢀⠌⠀⠨⢝⣿⠀⠘⡇⠀⠀⠀⠀
⢛⣧⠀⠀⠀⠀⢠⡇⢿⠀⢀⠪⣀⢔⠃⠀⠀⠀⠀⠀⢠⠋⢸⢻⠀⠀⢀⣤⣀⡀⠀⢠⣃⠀⠀⠀⠈⢄⡄⠀⢿⠀⠀⠀⠀
⠘⡈⠳⣄⣀⡰⢻⠀⢸⣤⡣⠊⢀⢆⠀⠀⠀⠀⠀⠀⡞⠀⠀⡈⠀⠇⣙⣿⣾⠇⡀⠘⡦⢕⢄⠀⠀⠀⠙⡄⢸⡇⠀⠀⠀
⡜⢽⠢⡀⣀⢴⡇⢡⢜⣯⡀⠀⡌⡁⠀⠀⠀⠀⠀⣸⠁⠀⠀⡇⢰⠀⢿⢨⢓⠒⡌⠜⣗⡀⠀⠑⢄⠀⠀⠰⡀⣧⠀⠀⠀
⠑⡑⠀⠀⢰⠕⠀⢸⠐⣽⠀⠀⢸⠁⠀⠀⠀⠀⢀⡏⠀⠀⢠⠁⢘⠀⠐⠋⠈⢰⠀⠀⠹⢏⢦⡀⠀⠱⡄⠀⢱⣻⠀⠀⠀
⠀⢱⡄⠀⠀⠀⠀⢸⡄⠻⡄⠀⠋⡀⠀⠀⠀⠀⣸⠁⠀⠀⢸⠀⠀⠑⠄⠠⠄⠂⠀⠀⡟⠈⢆⠉⠂⠀⠈⠳⣤⢻⡄⠀⠀
⠀⢸⡞⡄⠀⠀⠀⠘⡹⣦⣇⣠⢔⡇⠀⠀⠀⢠⢣⣶⣶⣶⡎⠀⠀⠘⠀⠀⠀⠀⠀⣸⠃⠀⠘⣆⠀⠀⠀⠀⠨⡺⡇⠀⠀
⠀⠈⣷⠈⢆⠀⠀⠀⠈⢸⡏⢴⠋⢠⠀⠀⠀⢢⣾⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠰⣿⠀⠀⠀⣿⢢⠀⠀⠀⠀⠙⣙⡆⠀
⠀⣿⡟⣧⢈⢆⠀⢰⣀⠔⡷⠁⡀⢼⠀⠀⠀⠘⣿⣿⣿⠟⡕⠀⠀⠀⠀⠀⠀⠀⡰⠘⠀⠀⣸⢹⣇⠃⠀⠀⠀⠀⠈⢣⠀
⢀⣿⠁⣿⢎⠘⠀⠈⠁⣼⠃⠀⠀⡧⠱⠀⠀⢀⢹⣽⠻⠊⠀⠀⠀⠀⠀⠀⠀⠠⣿⢰⠀⢠⠃⡸⡿⡠⡬⠖⣀⡀⠀⠀⠃
⢸⣿⡄⣿⠀⢻⡦⠠⣺⠟⠀⠀⠞⠀⠀⠀⠀⠈⠚⡗⠡⠀⠀⠀⠀⠀⠀⠀⠀⣷⣿⢸⠀⡎⠀⢧⠛⠉⠀⠀⠀⢀⠄⣠⠇
⣼⣿⣷⢹⠀⠈⣥⠈⠁⠀⠀⠀⡆⠀⠀⠀⠀⠀⠀⡗⡄⠀⠀⠀⠀⠀⠀⠀⡼⠋⠀⢻⣼⠃⣸⠜⠉⠀⣀⡤⠊⢀⠴⠋⠀
⣿⢻⡟⢸⡄⠀⠸⣧⠀⠀⠀⠀⣷⢦⣄⡀⠀⠀⠀⠃⡇⠀⠀⠀⠀⠀⠐⠊⠀⠀⢀⣸⣛⡔⠁⣀⠔⡘⠁⠀⢀⠕⠁⠀⠀'''


class Tui:
    """Owns the terminal layout for one interactive session."""

    def __init__(self):
        self._art = [line.rstrip() for line in ART.splitlines()]
        self._art_w = max(len(line) for line in self._art)
        self._real_stdout = None
        self._real_input = None
        self._shell = None
        self.enabled = False
        self.columns = 0
        self.rows = 0
        self.left_w = 0
        self.viewport_h = 0
        self._history = []
        self._pending = ''
        self._art_drawn = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
        if os.environ.get('TERM') == 'dumb':
            return False
        size = shutil.get_terminal_size()
        if size.columns < self._art_w + GAP + MIN_LEFT or size.lines < 10:
            return False
        self.enabled = True
        self._real_stdout = sys.stdout
        self._real_input = builtins.input
        self._shell = _Shell(self)
        sys.stdout = self._shell
        builtins.input = self._input
        self._raw('\x1b[2J\x1b[H')
        self._resize()  # draws the art (geometry always changed on first call)
        self._render()
        return True

    def stop(self):
        if not self.enabled:
            return
        self.enabled = False
        if self._pending:
            self._commit(self._pending)
            self._pending = ''
        sys.stdout = self._real_stdout
        builtins.input = self._real_input
        self._raw(f'\x1b[{self.rows};1H')

    # -- geometry ----------------------------------------------------------
    def _resize(self):
        size = shutil.get_terminal_size()
        if size.columns < self._art_w + GAP + MIN_LEFT or size.lines < 10:
            self._fallback()
            return
        changed = size.columns != self.columns or size.lines != self.rows
        self.columns = size.columns
        self.rows = size.lines
        self.left_w = size.columns - self._art_w - GAP
        self.viewport_h = max(1, self.rows - 2)
        if self.enabled and (changed or not self._art_drawn):
            # Redraw the art in its new position; the left column never
            # touches the art cells, so nothing else can move it.
            self._draw_art()

    def _fallback(self):
        """Terminal became too small: finish the session in plain mode."""
        if not self.enabled:
            return
        self.enabled = False
        if self._pending:
            self._commit(self._pending)
            self._pending = ''
        sys.stdout = self._real_stdout
        builtins.input = self._real_input
        self._real_stdout.write('\n')
        self._real_stdout.flush()

    # -- primitives --------------------------------------------------------
    def _raw(self, text):
        self._real_stdout.write(text)
        self._real_stdout.flush()

    @property
    def _art_col(self):
        return self.left_w + GAP + 1

    def _cell_left(self, row, text):
        # Pad with spaces instead of ESC[K: a clear-to-end-of-line would wipe
        # the art columns on the same row.
        text = text[:self.left_w]
        self._raw(f'\x1b[{row};1H{text.ljust(self.left_w)}')

    def _cell_art(self, row, text):
        self._raw(f'\x1b[{row};{self._art_col}H{text}\x1b[K')

    def _draw_art(self):
        for row, line in enumerate(self._art[:self.rows]):
            self._cell_art(row + 1, line)
        self._art_drawn = True

    def _wrap(self, text):
        if not text:
            return ['']
        return [text[i:i + self.left_w] for i in range(0, len(text), self.left_w)]

    def _commit(self, text):
        self._history.extend(self._wrap(text))
        if len(self._history) > 800:
            del self._history[:len(self._history) - 800]

    def _visible(self):
        lines = list(self._history)
        if self._pending:
            lines.extend(self._wrap(self._pending))
        return lines[-self.viewport_h:]

    def _render(self):
        if not self.enabled:
            return
        self._resize()
        if not self.enabled:
            return
        self._cell_left(1, f'dlss5-amd-hip {VERSION} — {TAGLINE}')
        visible = self._visible()
        for offset, line in enumerate(visible):
            self._cell_left(offset + 2, line)
        if visible:
            self._raw(f'\x1b[{len(visible) + 1};{len(visible[-1]) + 1}H')

    # -- input -------------------------------------------------------------
    def _input(self, prompt=''):
        if not self.enabled:
            return self._real_input(prompt)
        self._pending += prompt
        self._render()
        answer = self._real_input('')
        self._pending += answer
        self._commit(self._pending)
        self._pending = ''
        self._render()
        return answer


class _Shell:
    """File-like stdout that feeds the Tui line by line."""

    def __init__(self, tui: Tui):
        self._tui = tui

    def write(self, text):
        tui = self._tui
        if not tui.enabled:
            return tui._real_stdout.write(text)
        tui._pending += text
        while True:
            cuts = [tui._pending.index(char) + 1 for char in '\n\r' if char in tui._pending]
            if not cuts:
                break
            cut = min(cuts)
            tui._commit(tui._pending[:cut])
            tui._pending = tui._pending[cut:]
        tui._render()
        return len(text)

    def flush(self):
        if self._tui.enabled:
            self._tui._render()

    def isatty(self):
        return True

    def fileno(self):
        return self._tui._real_stdout.fileno()

    @property
    def encoding(self):
        return self._tui._real_stdout.encoding

    def close(self):
        pass
