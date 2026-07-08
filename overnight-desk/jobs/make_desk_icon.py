"""Generate the pixel-art app icon as a 512x512 PNG using only the stdlib.

Draws the desk's portfolio-manager sprite (dark suit, gold tie) on the UI's
panel background. Usage: uv run python jobs/make_desk_icon.py out.png
"""

from __future__ import annotations

import struct
import sys
import zlib

SPRITE = [
    "....hhhh....",
    "...hhhhhh...",
    "...ssssss...",
    "...sessse...",
    "...ssssss...",
    "....ssss....",
    "...bbtbbb...",
    "..bbbttbbb..",
    ".sbbbtbbbbs.",
    ".sbbbbbbbbs.",
    "..bbbbbbbb..",
    "...bbbbbb...",
    "...ll..ll...",
    "...ll..ll...",
    "..ff....ff..",
]
PALETTE = {
    "h": (0xC9, 0xAD, 0xA7),  # hair
    "s": (0xE8, 0xB9, 0x8A),  # skin
    "b": (0x22, 0x22, 0x3B),  # suit
    "t": (0xFF, 0xD1, 0x66),  # gold tie
    "e": (0x10, 0x10, 0x10),  # eyes
    "l": (0x22, 0x22, 0x3B),
    "f": (0x10, 0x10, 0x10),
}
BG = (0x13, 0x17, 0x29)
BORDER = (0xFF, 0xD1, 0x66)
SIZE = 512


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: str, pixels: list[list[tuple[int, int, int]]]) -> None:
    h, w = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + b"".join(bytes(p) for p in row) for row in pixels)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, 9))
        + png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def main(out: str) -> None:
    img = [[BG for _ in range(SIZE)] for _ in range(SIZE)]
    for i in range(SIZE):  # thin gold border, retro cartridge look
        for j in list(range(8)) + list(range(SIZE - 8, SIZE)):
            img[j][i] = BORDER
            img[i][j] = BORDER
    scale = 24
    ox = (SIZE - 12 * scale) // 2
    oy = (SIZE - 15 * scale) // 2
    for r, row in enumerate(SPRITE):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            color = PALETTE[ch]
            for dy in range(scale):
                for dx in range(scale):
                    img[oy + r * scale + dy][ox + c * scale + dx] = color
    write_png(out, img)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "desk_icon.png")
