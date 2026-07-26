#!/usr/bin/env python3
"""Generate placeholder portrait assets for PirateBot portrait mode.

This creates crude but functional PNG layers in
`portrait_viewer/assets/<set>/` so you can test the portrait player
before investing time in hand-crafted art. The generated pirate is a
stylized human (non-skeleton) pirate captain.

For a real show, replace these with AI-generated / hand-painted assets.

Usage:
    uv run python tools/generate_portrait_assets.py --output portrait_viewer/assets --set default
    uv run python tools/generate_portrait_assets.py --output portrait_viewer/assets --set captain_bob
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ASSET_NAMES = {
    "background",
    "body",
    "head",
    "mouth_rest",
    "mouth_ah",
    "mouth_ee",
    "mouth_oh",
    "mouth_f",
    "eye_left",
    "eye_right",
    "pupil",
}

WIDTH = 1280
HEIGHT = 720

# Layer layout shared with the browser player.
LAYOUT = {
    "canvas": {"width": WIDTH, "height": HEIGHT},
    "layers": ["background", "body", "head", "mouth", "eyes"],
    "head": {
        "center_x": 0.5,
        "center_y": 0.48,
        "width": 0.50,
    },
    "mouth": {
        "center_x": 0.5,
        "center_y": 0.59,
        "width": 0.18,
    },
    "eyes": {
        "left": {"center_x": 0.43, "center_y": 0.43, "width": 0.075, "height": 0.050},
        "right": {"center_x": 0.57, "center_y": 0.43, "width": 0.075, "height": 0.050},
    },
    "pupil": {"width": 0.028, "height": 0.038},
}


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
    """Torso, shoulders, coat, collar — no head."""
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

    # collar / shirt
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

    # painting texture overlay
    img = Image.alpha_composite(img, _noise_overlay(WIDTH, HEIGHT, intensity=8))
    return img


def generate_head() -> Image.Image:
    """Human pirate head with skin, hat, hair — no separate eyes/mouth cutouts."""
    img = _transparency()
    draw = ImageDraw.Draw(img)

    cx, cy = WIDTH // 2, int(HEIGHT * 0.55)

    # --- tricorn hat (behind hair/head) ---
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

    # --- scraggly hair behind ears ---
    hair_color = (140, 115, 80, 200)
    for dx in range(-110, 111, 22):
        x = cx + dx
        y = cy - 205
        draw.line(
            [(x, y), (x + np.random.randint(-18, 18), y + 45)],
            fill=hair_color,
            width=4,
        )

    # --- ears ---
    skin_dark = (185, 145, 115, 240)
    draw.ellipse([cx - 128, cy - 105, cx - 98, cy - 55], fill=skin_dark)
    draw.ellipse([cx + 98, cy - 105, cx + 128, cy - 55], fill=skin_dark)

    # --- head shape (weathered skin) ---
    head_color = (210, 170, 135, 255)
    head_box = [cx - 110, cy - 205, cx + 110, cy + 15]
    draw.ellipse(head_box, fill=head_color)

    # jaw/chin
    jaw_box = [cx - 82, cy - 15, cx + 82, cy + 78]
    draw.ellipse(jaw_box, fill=head_color)

    # --- facial features ---
    # nose
    draw.polygon(
        [(cx, cy - 70), (cx - 14, cy - 45), (cx + 14, cy - 45)],
        fill=(190, 140, 110, 240),
    )
    draw.arc([cx - 18, cy - 52, cx + 18, cy - 30], 200, 340, fill=(160, 120, 95, 200), width=4)

    # brow ridge shadows
    draw.arc([cx - 88, cy - 125, cx - 18, cy - 95], 200, 340, fill=(170, 130, 100, 180), width=8)
    draw.arc([cx + 18, cy - 125, cx + 88, cy - 95], 200, 340, fill=(170, 130, 100, 180), width=8)

    # bushy eyebrows
    brow_color = (110, 85, 60, 220)
    draw.arc([cx - 85, cy - 138, cx - 20, cy - 100], 200, 340, fill=brow_color, width=10)
    draw.arc([cx + 20, cy - 138, cx + 85, cy - 100], 200, 340, fill=brow_color, width=10)

    # closed/neutral eye creases (eye-white sprites go on top)
    draw.arc([cx - 75, cy - 98, cx - 28, cy - 72], 200, 340, fill=(170, 130, 100, 160), width=3)
    draw.arc([cx + 28, cy - 98, cx + 75, cy - 72], 200, 340, fill=(170, 130, 100, 160), width=3)

    # cheeks (rosy/tipsy)
    draw.ellipse([cx - 95, cy - 55, cx - 50, cy - 25], fill=(200, 130, 110, 90))
    draw.ellipse([cx + 50, cy - 55, cx + 95, cy - 25], fill=(200, 130, 110, 90))

    # scar across left cheek
    draw.line([(cx - 80, cy - 45), (cx - 35, cy - 20)], fill=(140, 90, 80, 150), width=3)

    # stubble
    stubble = (150, 120, 95, 80)
    for dx in range(-70, 71, 8):
        for dy in range(5, 60, 8):
            draw.point((cx + dx, cy + dy), fill=stubble)

    # closed mouth crease
    draw.line(
        [(cx - 45, cy + 28), (cx + 45, cy + 28)],
        fill=(155, 110, 95, 180),
        width=4,
    )

    # --- earring ---
    draw.ellipse([cx + 108, cy - 65, cx + 122, cy - 45], fill=(220, 200, 80, 230))
    draw.line([(cx + 115, cy - 45), (cx + 115, cy - 15)], fill=(220, 200, 80, 230), width=3)

    # --- painting texture overlay ---
    img = Image.alpha_composite(img, _noise_overlay(WIDTH, HEIGHT, intensity=8))
    return img


def _mouth_box(cx: int, cy: int) -> tuple[int, int, int, int]:
    # Smaller, lip-only region so the sprite can overlay the closed mouth on head.png
    return (cx - 45, cy + 20, cx + 45, cy + 55)


def generate_mouth(shape: str) -> Image.Image:
    """Mouth sprite only; everything outside the lips is transparent."""
    img = _transparency()
    draw = ImageDraw.Draw(img)
    cx, cy = WIDTH // 2, int(HEIGHT * 0.55)
    box = _mouth_box(cx, cy)

    lip_color = (175, 120, 105, 230)
    inner_color = (90, 45, 45, 220)
    teeth_color = (230, 225, 215, 230)

    if shape == "rest":
        draw.line(
            [(box[0] + 8, (box[1] + box[3]) // 2), (box[2] - 8, (box[1] + box[3]) // 2)],
            fill=lip_color,
            width=6,
        )
    elif shape == "ah":
        draw.ellipse(box, outline=lip_color, width=6)
        draw.ellipse(
            [box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10],
            fill=inner_color,
        )
    elif shape == "ee":
        grin_box = [box[0] - 10, box[1] + 2, box[2] + 10, box[3] - 2]
        draw.arc(grin_box, 0, 180, fill=lip_color, width=6)
        draw.line(
            [(box[0] + 6, box[1] + 6), (box[2] - 6, box[1] + 6)],
            fill=teeth_color,
            width=5,
        )
    elif shape == "oh":
        oh_box = [cx - 22, cy + 22, cx + 22, cy + 55]
        draw.ellipse(oh_box, outline=lip_color, width=6)
        draw.ellipse(
            [oh_box[0] + 7, oh_box[1] + 7, oh_box[2] - 7, oh_box[3] - 7],
            fill=inner_color,
        )
    elif shape == "f":
        teeth_box = [box[0] + 4, box[1] + 4, box[2] - 4, box[1] + 16]
        draw.rectangle(teeth_box, fill=teeth_color)
        draw.line(
            [(box[0] + 4, box[3] - 2), (box[2] - 4, box[3] - 2)],
            fill=lip_color,
            width=8,
        )

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


def generate_eyes() -> tuple[Image.Image, Image.Image]:
    """Eye-white sprites matching the head's eye sockets."""
    left = _transparency()
    right = _transparency()
    draw_l = ImageDraw.Draw(left)
    draw_r = ImageDraw.Draw(right)

    cx, cy = WIDTH // 2, int(HEIGHT * 0.55)
    eye_w, eye_h = 50, 36

    def draw_eye(draw, cx, cy):
        draw.ellipse(
            [cx - eye_w // 2, cy - eye_h // 2, cx + eye_w // 2, cy + eye_h // 2],
            fill=(245, 235, 220, 240),
            outline=(80, 60, 50, 120),
            width=2,
        )

    draw_eye(draw_l, cx - 50, cy - 85)
    draw_eye(draw_r, cx + 50, cy - 85)
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
    parser.add_argument(
        "--set",
        default="default",
        help="Asset set name / subfolder (default: default)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output) / args.set
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating placeholder portrait assets in {out_dir}...")

    generate_background().save(out_dir / "background.png")
    generate_body().save(out_dir / "body.png")
    generate_head().save(out_dir / "head.png")
    generate_head().save(out_dir / "head.png")

    for shape in ("rest", "ah", "ee", "oh", "f"):
        generate_mouth(shape).save(out_dir / f"mouth_{shape}.png")

    left_eye, right_eye = generate_eyes()
    left_eye.save(out_dir / "eye_left.png")
    right_eye.save(out_dir / "eye_right.png")
    generate_pupil().save(out_dir / "pupil.png")

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "name": args.set,
        "description": "Placeholder pirate portrait asset set",
        "layout": LAYOUT,
        "files": {name: f"assets/{args.set}/{name}.png" for name in sorted(ASSET_NAMES)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("Done. Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    print()
    print("These are placeholders. Replace them with hand-painted or AI-generated")
    print("art for the real Halloween show.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
