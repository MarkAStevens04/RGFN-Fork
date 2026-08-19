#!/usr/bin/env python
"""Seed every RNG, then hand off to REINVENT's own CLI unchanged.

WHY THIS EXISTS — an upstream bug, verified in v4.5.11, that silently costs you reproducibility.

``reinvent/Reinvent.py`` does:

    seed = input_config.get("seed", None)     # from the TOML
    if args.seed is not None:                 # ...but gated on the COMMAND-LINE flag
        set_seed(seed)                        # ...and passes the CONFIG value

and ``reinvent/validation.py``'s ``ReinventConfig`` is declared ``extra="forbid"`` with **no
``seed`` field**. So a TOML carrying ``seed = 42`` is rejected outright by pydantic, which means
``input_config.get("seed")`` can only ever be ``None``, which means ``set_seed(None)`` — and
``set_seed`` returns immediately on ``None``. The net effect: **no invocation of REINVENT 4.5.11
seeds its RNGs**, and it says nothing while failing to.

The consequence for this benchmark is specific and serious. Every cell is run at three seeds and
reported with error bars. Unseeded runs would still *differ* (torch seeds from entropy), so the
spread would be real — but the runs would not be reproducible, and "3 seeds" in a paper implies a
reader can regenerate them. This launcher restores that: it calls ``set_seed`` itself, before
REINVENT starts, and then defers to ``main_script()`` verbatim. REINVENT's own seeding code still
runs and is still a no-op; ours is what takes effect.

``PYTHONHASHSEED`` is deliberately NOT relied on from inside here. ``set_seed`` assigns it, but the
interpreter fixes its hash randomization at startup, so an in-process assignment cannot take effect —
the caller (``run_reinvent_fixed.py``) exports it into this process's environment instead, where it
does.

Usage (the seed is consumed, everything after it is REINVENT's own command line):
    python _seeded_launcher.py <seed> -l <logfile> <config.toml>
"""

import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <seed> [reinvent args...]")

    try:
        seed = int(sys.argv.pop(1))
    except ValueError as exc:
        raise SystemExit(f"first argument must be an integer seed, got {sys.argv[1]!r}") from exc

    from reinvent.utils.helpers import set_seed

    set_seed(seed)
    print(
        f"[seeded-launcher] set_seed({seed}) — REINVENT's own seeding is a no-op, see module doc",
        flush=True,
    )

    # sys.argv is now exactly what REINVENT's own console script would have seen.
    from reinvent.Reinvent import main_script

    main_script()


if __name__ == "__main__":
    main()
