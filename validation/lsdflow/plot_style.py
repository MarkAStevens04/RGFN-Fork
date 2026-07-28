"""Shared plot furniture for LSD-Flow figures — chiefly the "ideal direction" arrow.

Every figure we publish has an axis whose *good* direction is not self-evident: reactions/mode is
better LOW, modes-discovered is better HIGH, and a Pareto panel wants both at once. Worse, several of
our panels deliberately invert an axis (diversity sweeps run 0.9 -> 0.3 so "more diverse" reads
left-to-right; ``fixed_modes`` inverts y so "cheaper" is up), which makes the good direction genuinely
ambiguous to a reader seeing the plot for the first time. :func:`ideal_arrow` stamps it explicitly.

**Direction is in SCREEN space, not data space.** The arrow is drawn in axes-fraction coordinates, so
a caller with an inverted axis passes the direction the reader should look, not the direction the data
values increase. That is the whole point: on ``fixed_modes`` (inverted y, fewer reactions upward) the
correct call is ``ideal_arrow(ax, "up")``, even though the y *values* decrease upward.

Import contract: matplotlib only (no torch/rdkit), so it loads in every analysis env. Callers that
may run without matplotlib should still guard their own import; this module fails loudly if asked to
draw without it.

    from validation.lsdflow.plot_style import ideal_arrow
    ideal_arrow(ax, "up-right")                      # Pareto: more modes AND more diversity
    ideal_arrow(ax, "down", label="ideal (cheaper)") # cost panels
"""

from __future__ import annotations

# Unit screen-space vectors per direction name. Diagonals are normalized-ish so the arrow reads at
# the same visual length as the axis-aligned ones.
_DIRS = {
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up-right": (0.72, 0.72),
    "up-left": (-0.72, 0.72),
    "down-right": (0.72, -0.72),
    "down-left": (-0.72, -0.72),
}

# Unicode glyph appended to the label, so the direction survives greyscale printing and screen readers
# get something meaningful.
_GLYPH = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "up-right": "↗",
    "up-left": "↖",
    "down-right": "↘",
    "down-left": "↙",
}

# Anchor (axes-fraction) for the arrow's TAIL per requested location. Default keeps the annotation at
# the top of the panel — one consistent place to look across every figure — with the arrow itself
# pointing whichever way is good.
_LOCS = {
    "upper right": (0.90, 0.90),
    "upper left": (0.10, 0.90),
    "upper center": (0.50, 0.92),
    "lower right": (0.90, 0.14),
    "lower left": (0.10, 0.14),
}


def ideal_arrow(
    ax,
    direction: str,
    *,
    label: str = "ideal",
    loc: str = "upper right",
    length: float = 0.11,
    color: str = "#111111",
    alpha: float = 0.62,
    fontsize: float = 8.5,
    lw: float = 1.6,
):
    """Stamp a small "ideal direction" arrow inside ``ax``.

    Args:
        direction: screen-space direction (see :data:`_DIRS`) — ``"up"``, ``"down"``, ``"up-right"``,
            etc. NOT the direction the data values increase (see module docstring).
        label: text beside the arrow; the matching unicode glyph is appended automatically.
        loc: where the arrow sits, as an axes-fraction anchor (see :data:`_LOCS`). Pick one that
            misses the legend and the data — nothing here detects collisions.
        length: arrow length in axes fractions.

    Returns the annotation, so a caller can tweak/remove it.
    """
    if direction not in _DIRS:
        raise ValueError(f"direction must be one of {sorted(_DIRS)}, got {direction!r}")
    if loc not in _LOCS:
        raise ValueError(f"loc must be one of {sorted(_LOCS)}, got {loc!r}")
    dx, dy = _DIRS[direction]
    x0, y0 = _LOCS[loc]
    # Draw from the anchor toward the ideal corner, clamped into the axes so a diagonal never spills.
    x1 = min(max(x0 + dx * length, 0.02), 0.98)
    y1 = min(max(y0 + dy * length, 0.02), 0.98)
    ann = ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            alpha=alpha,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=13,
        ),
        annotation_clip=False,
        zorder=6,
    )
    if label:
        # Offset the text opposite the arrow so it never sits under the shaft, then clamp it inside
        # the axes. Alignment is chosen from where the label ACTUALLY lands, not from the arrow
        # direction: anchoring at x=0.90 and aligning left pushes the text off the right edge (it
        # clipped "ideal ↖" on the count-once curve before this).
        ox, oy = -dx * 0.055, -dy * 0.055
        if dy == 0.0:  # a horizontal arrow: offsetting along -x lands the text ON the shaft
            ox, oy = -dx * 0.015, -0.075  # so drop it below instead
        xl = min(max(x0 + ox, 0.04), 0.96)
        yl = min(max(y0 + oy, 0.04), 0.96)
        ax.text(
            xl,
            yl,
            f"{label} {_GLYPH[direction]}",
            transform=ax.transAxes,
            ha="right" if xl > 0.5 else "left",
            va="top" if yl > 0.5 else "bottom",
            fontsize=fontsize,
            color=color,
            alpha=alpha,
            style="italic",
            zorder=6,
        )
    return ann
