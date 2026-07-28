"""Shared plot furniture for LSD-Flow figures — the "ideal direction" marker.

Every figure we publish has a metric whose *good* direction is not self-evident: reactions/mode is
better LOW, modes-discovered is better HIGH, and a Pareto panel wants both at once. Several panels also
deliberately invert an axis (diversity sweeps run 0.9 -> 0.3 so "more diverse" reads left-to-right;
``fixed_modes`` inverts y so "cheaper" is up), which makes the good direction genuinely ambiguous to a
first-time reader.

**Convention: a parenthesised arrow appended to the panel TITLE, never drawn inside the axes.** This is
standard publication practice ("FID ↓", "Accuracy ↑") — it costs no plot area, cannot collide with the
data or the legend, survives cropping, and reads correctly in a figure list or a caption. Do not add
arrows, shaded "better" regions, or corner annotations to the plotting area.

**The glyph describes the METRIC, not the screen.** ``↓`` always means "lower values are better",
regardless of whether the axis happens to be inverted. That is what makes it unambiguous on our flipped
panels: ``fixed_modes`` plots reactions on an inverted y-axis, and its marker is still ``(↓)`` because
fewer reactions is better.

    from validation.lsdflow.plot_style import ideal_marker

    ax.set_title(f"cost per mode {ideal_marker('lower')}")            # -> "... (↓)"
    ax.set_title(f"modes found {ideal_marker('higher')}")             # -> "... (↑)"
    ax.set_title("count-once curve " + ideal_marker(                  # two metrics at once
        ("higher", "modes"), ("lower", "reactions")))                 # -> "(↑ modes, ↓ reactions)"

Import contract: pure stdlib (no matplotlib, no torch), so it loads in every analysis env and in tests.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

# "lower"/"higher" describe the METRIC's good direction, never a screen direction.
_GLYPH = {"lower": "↓", "higher": "↑", "down": "↓", "up": "↑"}

Spec = Union[str, Tuple[str, str]]


def _one(spec: Spec) -> str:
    if isinstance(spec, str):
        direction, name = spec, ""
    else:
        direction, name = spec
    key = direction.strip().lower()
    if key not in _GLYPH:
        raise ValueError(f"direction must be 'lower' or 'higher', got {direction!r}")
    return f"{_GLYPH[key]} {name}".strip()


def ideal_marker(*specs: Spec) -> str:
    """The parenthesised ideal-direction marker to append to a panel title.

    Args:
        *specs: one entry per metric. Either ``"lower"``/``"higher"``, or a
            ``(direction, metric_name)`` pair when a panel has more than one metric and the arrows
            would otherwise be ambiguous.

    Returns:
        e.g. ``"(↓)"``, ``"(↑)"``, ``"(↑ modes, ↓ reactions)"``. Empty string if no specs are given,
        so a caller can pass through unconditionally.
    """
    # Drop empty/None entries so a caller can pass an optional value straight through without
    # branching (a bare `ideal or ""` must not raise).
    kept = [s for s in specs if s]
    if not kept:
        return ""
    return "(" + ", ".join(_one(s) for s in kept) + ")"


def title_with_ideal(title: str, *specs: Spec) -> str:
    """``title`` with the marker appended — the one-call form for ``ax.set_title``."""
    marker = ideal_marker(*specs)
    return f"{title} {marker}" if marker else title


def axis_label_with_ideal(label: str, direction: str) -> str:
    """A single-metric axis label carrying its own marker, e.g. ``"reactions per mode (↓)"``.

    Use when a panel's title covers something else (a question, a cell name) and the direction belongs
    to the axis. Still text — nothing is drawn inside the axes.
    """
    return f"{label} {ideal_marker(direction)}"


def describe(*specs: Sequence[Spec]) -> str:
    """Long form for captions/logs: ``"lower is better"`` / ``"higher is better"``."""
    parts = []
    for spec in specs:
        direction, name = (spec, "") if isinstance(spec, str) else spec
        word = "lower" if _GLYPH[direction.strip().lower()] == "↓" else "higher"
        parts.append(f"{word} {name}".strip() + " is better")
    return "; ".join(parts)
