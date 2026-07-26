#!/usr/bin/env python3
"""Generate placeholder portrait assets for PirateBot portrait mode.

This creates crude but functional PNG layers in
`portrait_viewer/assets/` so you can test the portrait player before
investing time in hand-crafted art. The generated pirate is a stylized
skeleton with simple geometric mouth/eye shapes.

For a real show, replace these with AI-generated / hand-painted assets.

Usage:
    uv run python tools/generate_portrait_assets.py --output portrait_viewer/assets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ASSET_NAMES = {
    "background",
    "body",
    "mouth_rest",
    "mouth_ah",
    "mouth_ee",
    "mouth_oh",
    "mouth_f",
}

WIDTH = 1280
HEIGHT = 720


def _transparency() -> Image.Image:
    return Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))


def _noise_overlay(width: int, height: int, intensity: int = 12) -> Image.Image:
    """Subtle canvas grain."""
    arr = np.random.randint(-intensity, intensity + 1, (height, width, 4), dtype=np.int16)
    arr[:, :, 3] = 16  # low alpha
    img = Image.fromarray((arr + 128).astype(np.uint8), "RGBA")
    return img


def _vignette(width: int, height: int) -> Image.Image:
    base = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(base)
    cx, cy = width // 2, height // 2
    max_r = int((cx ** 2 + cy ** 2) ** 0.5)
    for r in range(max_r, 0, -5):
        alpha = int(100 * (1 - (r / max_r) ** 1.5))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=alpha)
    return base.convert("RGBA")


def generate_background() -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), (15, 10, 8, 255))
    draw = ImageDraw.Draw(img)

    # stormy gradient sky
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(25 - 15 * t)
        g = int(18 - 10 * t)
        b = int(20 - 12 * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b, 255))

    # moon
    moon_x, moon_y = WIDTH * 0.75, HEIGHT * 0.25
    for r in range(80, 0, -2):
        alpha = int(200 * (r / 80))
        draw.ellipse(
            [moon_x - r, moon_y - r, moon_x + r, moon_y + r],
            fill=(255, 240, 200, alpha),
        )

    # rough sea
    for i in range(8):
        y = int(HEIGHT * (0.55 + i * 0.06))
        wave_color = (10 + i * 3, 15 + i * 2, 25 + i * 3, 200)
        draw.line([(0, y), (WIDTH, y)], fill=wave_color, width=18 + i * 3)

    # ship silhouette
    ship_pts = [
        (WIDTH * 0.12, HEIGHT * 0.58),
        (WIDTH * 0.22, HEIGHT * 0.58),
        (WIDTH * 0.20, HEIGHT * 0.50),
        (WIDTH * 0.14, HEIGHT * 0.50),
    ]
    draw.polygon(ship_pts, fill=(8, 6, 5, 220))
    draw.line(
        [(WIDTH * 0.18, HEIGHT * 0.50), (WIDTH * 0.18, HEIGHT * 0.40)],
        fill=(8, 6, 5, 200),
        width=4,
    )

    img = Image.alpha_composite(img, _noise_overlay(WIDTH, HEIGHT))
    vignette = _vignette(WIDTH, HEIGHT)
    img = Image.alpha_composite(img, vignette)
    return img


def generate_body() -> Image.Image:
    img = _transparency()
    draw = ImageDraw.Draw(img)

    cx, cy = WIDTH // 2, int(HEIGHT * 0.55)

    # tattered coat shoulders
    coat_color = (35, 45, 65, 230)
    draw.polygon(
        [
            (cx - 260, HEIGHT),
            (cx - 180, cy - 40),
            (cx - 80, cy + 20),
            (cx + 80, cy + 20),
            (cx + 180, cy - 40),
            (cx + 260, HEIGHT),
        ],
        fill=coat_color,
    )

    # collar
    draw.polygon(
        [
            (cx - 80, cy - 10),
            (cx - 50, cy - 90),
            (cx, cy - 70),
            (cx + 50, cy - 90),
            (cx + 80, cy - 10),
        ],
        fill=(50, 55, 70, 230),
    )

    # skull head
    head_color = (220, 215, 205, 255)
    head_box = [cx - 110, cy - 230, cx + 110, cy + 30]
    draw.ellipse(head_box, fill=head_color)

    # jaw
    jaw_box = [cx - 80, cy - 20, cx + 80, cy + 90]
    draw.ellipse(jaw_box, fill=head_color)

    # cheek hollows
    draw.ellipse([cx - 95, cy - 80, cx - 45, cy - 30], fill=(180, 170, 160, 180))
    draw.ellipse([cx + 45, cy - 80, cx + 95, cy - 30], fill=(180, 170, 160, 180))

    # nose hole
    draw.polygon(
        [(cx, cy - 55), (cx - 12, cy - 35), (cx + 12, cy - 35)],
        fill=(30, 25, 25, 220),
    )

    # empty eye sockets (we'll overlay animated eyes)
    draw.ellipse([cx - 75, cy - 110, cx - 30, cy - 70], fill=(20, 15, 15, 240))
    draw.ellipse([cx + 30, cy - 110, cx + 75, cy - 70], fill=(20, 15, 15, 240))

    # eyebrows
    draw.arc([cx - 85, cy - 135, cx - 20, cy - 95], 200, 340, fill=(60, 50, 45, 220), width=5)
    draw.arc([cx + 20, cy - 135, cx + 85, cy - 95], 200, 340, fill=(60, 50, 45, 220), width=5)

    # tricorn hat
    hat_color = (40, 35, 30, 240)
    draw.polygon(
        [
            (cx - 200, cy - 170),
            (cx - 100, cy - 240),
            (cx + 100, cy - 240),
            (cx + 200, cy - 170),
            (cx + 80, cy - 180),
            (cx, cy - 300),
            (cx - 80, cy - 180),
        ],
        fill=hat_color,
    )
    draw.ellipse([cx - 120, cy - 200, cx + 120, cy - 160], fill=hat_color)

    # scraggly hair
    hair_color = (180, 170, 150, 180)
    for dx in range(-90, 91, 20):
        x = cx + dx
        y = cy - 210
        draw.line([(x, y), (x + np.random.randint(-15, 15), y + 40)], fill=hair_color, width=3)

    # earring
    draw.ellipse([cx + 108, cy - 70, cx + 122, cy - 50], fill=(220, 200, 80, 230))
    draw.line([(cx + 115, cy - 50), (cx + 115, cy - 20)], fill=(220, 200, 80, 230), width=3)

    # painting texture overlay
    img = Image.alpha_composite(img, _noise_overlay(WIDTH, HEIGHT, intensity=8))
    return img


def _mouth_box(cx: int, cy: int) -> tuple[int, int, int, int]:
    return (cx - 50, cy + 10, cx + 50, cy + 55)


def generate_mouth(shape: str) -> Image.Image:
    img = _transparency()
    draw = ImageDraw.Draw(img)
    cx, cy = WIDTH // 2, int(HEIGHT * 0.55)
    box = _mouth_box(cx, cy)

    # dark mouth cavity behind lips
    draw.ellipse(box, fill=(30, 20, 20, 200))

    lip_color = (165, 140, 130, 230)
    inner_color = (90, 50, 50, 220)

    if shape == "rest":
        # thin closed line
        draw.line(
            [(box[0] + 10, (box[1] + box[3]) // 2), (box[2] - 10, (box[1] + box[3]) // 2)],
            fill=lip_color,
            width=6,
        )
    elif shape == "ah":
        # big open oval
        draw.ellipse(box, outline=lip_color, width=6)
        draw.ellipse(
            [box[0] + 12, box[1] + 12, box[2] - 12, box[3] - 12],
            fill=inner_color,
        )
    elif shape == "ee":
        # wide stretched grin
        grin_box = [box[0] - 10, box[1] + 5, box[2] + 10, box[3] - 5]
        draw.arc(grin_box, 0, 180, fill=lip_color, width=6)
        draw.arc(
            [grin_box[0] + 8, grin_box[1] + 8, grin_box[2] - 8, grin_box[3] - 8],
            0,
            180,
            fill=inner_color,
            width=4,
        )
    elif shape == "oh":
        # small round "o"
        oh_box = [cx - 25, cy + 15, cx + 25, cy + 55]
        draw.ellipse(oh_box, outline=lip_color, width=6)
        draw.ellipse(
            [oh_box[0] + 8, oh_box[1] + 8, oh_box[2] - 8, oh_box[3] - 8],
            fill=inner_color,
        )
    elif shape == "f":
        # bottom lip under teeth
        teeth_box = [box[0] + 5, box[1] + 5, box[2] - 5, box[1] + 20]
        draw.rectangle(teeth_box, fill=(230, 230, 220, 230))
        draw.line(
            [(box[0] + 5, box[3] - 5), (box[2] - 5, box[3] - 5)],
            fill=lip_color,
            width=8,
        )

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


def generate_eyes() -> tuple[Image.Image, Image.Image]:
    """Generate optional eye-white images matching the skull sockets."""
    left = _transparency()
    right = _transparency()
    draw_l = ImageDraw.Draw(left)
    draw_r = ImageDraw.Draw(right)

    cx, cy = WIDTH // 2, int(HEIGHT * 0.55)
    eye_w, eye_h = 45, 40

    def draw_eye(draw, cx, cy):
        draw.ellipse(
            [cx - eye_w // 2, cy - eye_h // 2, cx + eye_w // 2, cy + eye_h // 2],
            fill=(240, 230, 215, 230),
            outline=(30, 25, 25, 120),
            width=2,
        )

    draw_eye(draw_l, cx - 52, cy - 90)
    draw_eye(draw_r, cx + 52, cy - 90)
    return left, right


def generate_pupil() -> Image.Image:
    size = 24
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([2, 2, size - 2, size - 2], fill=(25, 20, 18, 240))
    draw.ellipse([6, 6, 10, 10], fill=(200, 190, 170, 180))
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "portrait_viewer" / "assets"),
        help="Directory to write PNG assets",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating placeholder portrait assets in {out_dir}...")

    generate_background().save(out_dir / "background.png")
    generate_body().save(out_dir / "body.png")

    for shape in ("rest", "ah", "ee", "oh", "f"):
        generate_mouth(shape).save(out_dir / f"mouth_{shape}.png")

    left_eye, right_eye = generate_eyes()
    left_eye.save(out_dir / "eye_left.png")
    right_eye.save(out_dir / "eye_right.png")
    generate_pupil().save(out_dir / "pupil.png")

    print("Done. Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    print()
    print("These are placeholders. Replace them with hand-painted or AI-generated")
    print("art for the real Halloween show.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
