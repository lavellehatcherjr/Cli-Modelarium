"""Startup banner for cli-modelarium.

Shown only on bare `cli-modelarium` invocation in an interactive terminal.
Rendered to stderr so it can never contaminate stdout data pipelines
(JSON/CSV/Markdown output that users pipe to jq, redirect to files, or
parse in CI).

ASCII art is hardcoded (generated once with the figlet 'standard' font -
no runtime dependency on pyfiglet). The gradient uses rich, which is
already a dependency.

Colors are the Tokyo Night brand palette matching the cli-modelarium logo:
  #7AA2F7 primary blue, #BB9AF7 accent purple.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.text import Text

# "Cli Modelarium" rendered with the figlet 'standard' font.
# Generated once during development; hardcoded here (no pyfiglet at runtime).
_BANNER_ART = r"""
  ____ _ _   __  __           _      _            _
 / ___| (_) |  \/  | ___   __| | ___| | __ _ _ __(_)_   _ _ __ ___
| |   | | | | |\/| |/ _ \ / _` |/ _ \ |/ _` | '__| | | | | '_ ` _ \
| |___| | | | |  | | (_) | (_| |  __/ | (_| | |  | | |_| | | | | | |
 \____|_|_| |_|  |_|\___/ \__,_|\___|_|\__,_|_|  |_|\__,_|_| |_| |_|
"""

# Tokyo Night brand palette (matches the logo wordmark).
_BLUE = (0x7A, 0xA2, 0xF7)    # #7AA2F7
_PURPLE = (0xBB, 0x9A, 0xF7)  # #BB9AF7


def _lerp(a: int, b: int, t: float) -> int:
    """Linear interpolation between two ints."""
    return int(a + (b - a) * t)


def _gradient_line(
    line: str,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> Text:
    """Apply a horizontal RGB gradient across one line of text."""
    text = Text()
    n = max(len(line) - 1, 1)
    for i, ch in enumerate(line):
        t = i / n
        r = _lerp(start[0], end[0], t)
        g = _lerp(start[1], end[1], t)
        b = _lerp(start[2], end[2], t)
        text.append(ch, style=f"rgb({r},{g},{b})")
    return text


def should_show_banner() -> bool:
    """Return True only when the banner is safe to show.

    The banner is safe only in an interactive terminal. When stdout is
    piped, redirected, or running in CI, isatty() is False and we skip
    the banner so it can never interfere with output or scripting.
    """
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        # Defensive: if stdout is unusual (closed, replaced), don't show.
        return False


def render_banner() -> None:
    """Render the cli-modelarium banner to stderr with a brand gradient.

    Emitted to stderr (not stdout) for defense-in-depth: even if this were
    ever called outside the bare-invocation branch, it physically cannot
    land in a stdout data pipeline.

    rich's Console respects NO_COLOR automatically; on no-color or legacy
    terminals the gradient degrades to plain text gracefully.
    """
    console = Console(stderr=True)
    console.print()
    for line in _BANNER_ART.strip("\n").splitlines():
        console.print(_gradient_line(line, _BLUE, _PURPLE))
    console.print()
    tagline = Text()
    tagline.append(
        "  Statistically rigorous LLM comparison",
        style="bold rgb(122,162,247)",
    )
    console.print(tagline)
    console.print("  climodelarium.com", style="rgb(187,154,247)")
    console.print()
