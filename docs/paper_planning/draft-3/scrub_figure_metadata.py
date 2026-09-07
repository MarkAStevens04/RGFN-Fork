#!/usr/bin/env python3
"""Strip identifying metadata from a figure PDF before it is embedded in the manuscript.

WHY THIS EXISTS. pdfTeX carries an included figure's own metadata into the output. The
Canva export of the method diagram contained

    /Author (Mark Stevens)  /Creator (Canva)  /Producer (Canva)
    /Title (Hub-Batching Method Diagram)  /Keywords (<two Canva design ids>)

and all of it ended up inside main.pdf -- while main.pdf's own /Info stayed empty, so
checking the obvious place would not have caught it. For a double-blind submission that is
a real leak: `strings main.pdf | grep -i author` finds it in seconds.

USAGE
    python scrub_figure_metadata.py figmethod.pdf          # scrub in place (writes a .bak)
    python scrub_figure_metadata.py --check main.pdf       # scan only, change nothing

Values are blanked in place and padded to the original byte length, so no cross-reference
offset moves and the file stays valid without needing a full rewrite.

CAVEAT, and it is why --check exists: if a producer put the metadata inside a COMPRESSED
object stream, in-place blanking cannot reach it. The script says so rather than reporting
success. In that case re-export the figure as PNG -- raster formats do not carry an /Author
field, and for a line diagram the quality cost is nil at 2-3x scale.
"""
import re
import shutil
import sys
import zlib

FIELDS = (b"Author", b"Creator", b"Producer", b"Keywords", b"Title", b"Subject")


def scan(data):
    """Return (uncompressed hits, compressed hits). Compressed ones cannot be blanked."""
    plain, packed = [], []
    for f in FIELDS:
        for m in re.finditer(rb"/" + f + rb"\s*\(([^)]{1,300})\)", data):
            plain.append((f.decode(), m.group(1).decode("latin-1", "replace"), m.start()))
    for m in re.finditer(rb"stream\r?\n", data):
        s = m.end()
        e = data.find(b"endstream", s)
        if e < 0:
            continue
        try:
            t = zlib.decompress(data[s:e])
        except Exception:
            continue
        for f in FIELDS:
            for mm in re.finditer(rb"/" + f + rb"\s*\(([^)]{1,300})\)", t):
                packed.append((f.decode(), mm.group(1).decode("latin-1", "replace")))
    return plain, packed


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    if not args:
        sys.exit(__doc__)
    path = args[0]
    data = open(path, "rb").read()

    plain, packed = scan(data)
    if not plain and not packed:
        print(f"  {path}: no identifying metadata found")
        return 0

    for k, v, _ in plain:
        print(f"  uncompressed  /{k:<9} ({v[:70]})")
    for k, v in packed:
        print(f"  COMPRESSED    /{k:<9} ({v[:70]})   <-- cannot blank in place")

    if check_only:
        print("\n  --check: nothing written")
        return 1 if (plain or packed) else 0

    shutil.copy(path, path + ".bak")
    out = bytearray(data)
    for k, v, off in plain:
        m = re.match(rb"/" + k.encode() + rb"\s*\(([^)]*)\)", bytes(out[off : off + 400]))
        if not m:
            continue
        a, b = off + m.start(1), off + m.end(1)
        out[a:b] = b" " * (b - a)  # same length: no xref offsets move
    open(path, "wb").write(bytes(out))

    plain2, packed2 = scan(bytes(out))
    left = [(k, v) for k, v, _ in plain2 if v.strip()] + [(k, v) for k, v in packed2 if v.strip()]
    print(f"\n  wrote {path} (backup at {path}.bak)")
    if left:
        print("  STILL PRESENT after scrubbing -- re-export the figure as PNG instead:")
        for k, v in left:
            print(f"    /{k} ({v[:70]})")
        return 1
    print("  verified clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
