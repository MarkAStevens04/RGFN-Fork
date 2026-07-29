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

Two flavours, and picking the right one matters for readability:

* **One objective** -> :func:`ideal_marker` gives an axis-aligned ``(↓)`` / ``(↑)``. The glyph describes
  the METRIC, not the screen: ``↓`` always means "lower values are better" even where the axis is
  flipped, so ``fixed_modes`` (reactions on an inverted y) is correctly ``(↓)``.
* **Both axes are objectives** (every Pareto panel) -> :func:`pareto_marker` gives a single DIAGONAL
  ``(↗)`` pointing at the desirable corner. Prefer this over two arrows: ``(↑ modes, ↓ reactions)``
  makes the reader resolve two axes before judging a curve, whereas one diagonal says "we want to be
  over here, and here is how close each model gets" immediately. Because a diagonal names a *corner*,
  it is the one case that must know about axis inversion — so its API takes the metric directions plus
  the ``invert_*`` flags and derives the corner itself.

    from validation.lsdflow.plot_style import ideal_marker, pareto_marker

    ax.set_title(f"cost per mode {ideal_marker('lower')}")             # -> "... (↓)"
    ax.set_title(f"modes found {ideal_marker('higher')}")              # -> "... (↑)"
    ax.set_title("count-once curve " + pareto_marker(                  # x = reactions, y = modes
        x="lower", y="higher"))                                        # -> "(↖)"
    ax.set_title("diversity pareto " + pareto_marker(                  # x flipped: 0.9 -> 0.3
        x="lower", y="higher", invert_x=True))                         # -> "(↗)"

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


# Screen corner -> glyph, keyed by (good direction is rightward?, good direction is upward?).
_CORNER = {
    (True, True): "↗",
    (False, True): "↖",
    (True, False): "↘",
    (False, False): "↙",
}


def pareto_marker(
    *, x: str, y: str, invert_x: bool = False, invert_y: bool = False, note: str = ""
) -> str:
    """A single DIAGONAL marker for a panel where both axes are objectives — ``"(↗)"``.

    Use this instead of ``ideal_marker("higher", "lower")`` on Pareto-style panels. Two arrows make a
    reader resolve "up on this axis, down on that one" before they can judge a curve; one diagonal
    says "we want to be in this corner — here is how close each model gets" at a glance.

    Unlike the axis-aligned markers, a diagonal necessarily names a *corner of the plot*, so it must
    account for inverted axes. The API stays METRIC-based — you say which direction is good for the
    quantity on each axis, plus whether that axis is flipped — and the corner is derived. Never
    hand-pick a glyph: on our diversity sweeps (x runs 0.9 -> 0.3) the naive choice is backwards.

    Args:
        x: good direction for the x-axis quantity — ``"lower"`` or ``"higher"``.
        y: good direction for the y-axis quantity.
        invert_x: pass ``True`` if the caller called ``ax.invert_xaxis()``.
        invert_y: likewise for the y-axis.
        note: optional short gloss placed after the glyph, e.g. ``"more diverse, more modes"``.

    Example — the diversity Pareto (x = Tanimoto cutoff, inverted, lower = more diverse; y = modes
    discovered, higher better) resolves to ``"(↗)"``::

        pareto_marker(x="lower", y="higher", invert_x=True)

    And the count-once curve (x = reactions spent, y = modes gained, neither inverted) to ``"(↖)"``.
    """
    for name, val in (("x", x), ("y", y)):
        if val.strip().lower() not in _GLYPH:
            raise ValueError(f"{name} must be 'lower' or 'higher', got {val!r}")
    # "Good is rightward" iff the good direction is toward larger x AND x is not flipped (or the
    # reverse of both). Same for up.
    rightward = (_GLYPH[x.strip().lower()] == "↑") != bool(invert_x)
    upward = (_GLYPH[y.strip().lower()] == "↑") != bool(invert_y)
    glyph = _CORNER[(rightward, upward)]
    return f"({glyph} {note})" if note else f"({glyph})"


def title_with_pareto_ideal(title: str, **kw) -> str:
    """``title`` with a diagonal Pareto marker appended (see :func:`pareto_marker`)."""
    return f"{title} {pareto_marker(**kw)}"


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
