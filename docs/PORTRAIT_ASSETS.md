# Portrait Asset Specification

This document defines the PNG layers used by PirateBot's "haunted painting"
portrait mode. The goal is to make it easy to swap pirates, backgrounds,
and moods by changing one config value, then iterating on art without
touching code.

## Quick start

Generate a testable placeholder set:

```bash
uv run python tools/generate_portrait_assets.py \
  --output portrait_viewer/assets \
  --set default
```

Swap to another set later:

1. Generate a new set:
   ```bash
   uv run python tools/generate_portrait_assets.py \
     --output portrait_viewer/assets \
     --set captain_bob
   ```
2. Edit `config.portrait.yaml`:
   ```yaml
   portrait:
     asset_set: "captain_bob"
   ```
3. Restart `main.py`. The browser reloads `assets/captain_bob/manifest.json`
   automatically.

## Canvas

All PNGs must share the same pixel dimensions. The default is **1280×720**
(16:9). If you want a different aspect ratio or resolution, edit the
`layout.canvas` block in `manifest.json` and regenerate.

## Required layers

| File | Transparency | Contents | Notes |
|------|-------------|----------|-------|
| `background.png` | Opaque | Full scene behind the pirate. | Sky, moon, ship, sea, frame interior, etc. |
| `body.png` | Alpha | Torso, shoulders, coat, shirt. | No head. Positioned so the neck meets the head layer. |
| `head.png` | Alpha | Full head with skin, hat, hair, ears. | No separate eyes or mouth cutouts. Paint a closed/neutral mouth and relaxed eye sockets; eyes and mouth are overlaid at runtime. |
| `mouth_rest.png` | Alpha | Closed lips only. | Small region, transparent outside lips. |
| `mouth_ah.png` | Alpha | Open wide mouth. | For loud/exaggerated sounds. |
| `mouth_ee.png` | Alpha | Wide smile / teeth showing. | For "ee" / grin sounds. |
| `mouth_oh.png` | Alpha | Small round "o". | For rounded vowels. |
| `mouth_f.png` | Alpha | Bottom lip under teeth. | For "f" / "v" sounds. |
| `eye_left.png` | Alpha | Left eye white. | Should match the closed eye socket painted on `head.png`. |
| `eye_right.png` | Alpha | Right eye white. | Mirror of `eye_left.png`. |
| `pupil.png` | Alpha | Single pupil / iris. | Centered in a small square; scaled to fit inside the eye whites. |

## Manifest

Every asset set contains a `manifest.json` describing layer positions:

```json
{
  "name": "default",
  "description": "...",
  "layout": {
    "canvas": {"width": 1280, "height": 720},
    "layers": ["background", "body", "head", "mouth", "eyes"],
    "head": {
      "center_x": 0.5,
      "center_y": 0.48,
      "width": 0.50
    },
    "mouth": {
      "center_x": 0.5,
      "center_y": 0.59,
      "width": 0.18
    },
    "eyes": {
      "left": {"center_x": 0.43, "center_y": 0.43, "width": 0.075, "height": 0.050},
      "right": {"center_x": 0.57, "center_y": 0.43, "width": 0.075, "height": 0.050}
    },
    "pupil": {"width": 0.028, "height": 0.038}
  },
  "files": { "background": "assets/default/background.png", ... }
}
```

All coordinates are **normalized 0–1** relative to the canvas width/height.
This lets the browser position layers correctly regardless of monitor size.

### Layout tuning

If your generated or painted head sits higher/lower than the placeholder:

1. Open `http://localhost:9877/`.
2. Edit `portrait_viewer/assets/<set>/manifest.json`.
3. Adjust `head.center_y`, `mouth.center_y`, and `eyes.*.center_*`.
4. Hard-refresh the browser (`Ctrl+Shift+R`).

You do **not** need to restart the Python backend for layout changes.

## Alpha rules

- **Background** can be opaque.
- **Body, head, and mouth sprites** must use alpha transparency everywhere
  except the painted pixels.
- **Mouth sprites** should be a small region around the lips. Do not include
  cheeks, chin, or beard — those stay on `head.png`.
- **Eye whites** should be tightly cropped to the eye shape. The pupil is
  constrained inside this region at runtime.

## Art style tips for iteration

For a coherent "haunted oil painting" look:

1. Use the same canvas size for every layer.
2. Keep lighting direction consistent (e.g. warm candlelight from below-left).
3. Mute saturation slightly and add a subtle canvas texture/noise.
4. Make the pirate look *at* the viewer (camera), not off-camera.
5. Leave eye sockets neutral/closed on `head.png` so the animated eyes read
   clearly on top.

## Example prompts for AI image generation

These are starting points you can paste into ComfyUI / Flux / SD:

### Background
> Framed oil painting of a stormy moonlit sea at night, a ghost galleon on
> the horizon, dark clouds, cinematic, dramatic lighting, vintage canvas
> texture, 16:9.

### Pirate head ( Captain Barnacle Bill )
> Oil-painted portrait of a grizzled human pirate captain, weathered skin,
> bushy eyebrows, tricorn hat, scraggly hair, single gold earring, small scar
> on cheek, warm candlelight from below, looking at viewer, neutral closed
> mouth, transparent background, 16:9 canvas.

### Pirate body
> Oil-painted torso of a pirate captain in a tattered navy coat with brass
> buttons, white ruffled collar, broad shoulders, transparent background,
> 16:9 canvas.

### Mouth set
Generate one clean closed mouth, then paint variants for ah/ee/oh/f. Keep
only the lip region; erase everything else to transparency.

> Oil-painted pirate mouth, [closed / wide open / wide grin showing teeth /
> small round o / bottom lip under teeth], neutral expression, transparent
> background, 16:9 canvas.

## Future animation

The current code already groups body + head + mouth + eyes into a single
`#avatar` wrapper for whole-body emotes, and the head has its own wrapper
(`#head-wrap`) for future tilt/turn classes. To add simple tweening later:

- Add CSS classes like `tilt-left`, `tilt-right`, `nod`, `shake` to
  `#head-wrap`.
- Have the Python backend emit `play_animation` commands, or drive them from
  `portrait.js` based on gaze/idle state.
- For more advanced animation, split `head.png` further into `head_base.png`,
  `beard.png`, `hat.png`, etc.

## Directory layout

```
portrait_viewer/
├── index.html
├── portrait.css
├── portrait.js
└── assets/
    ├── default/
    │   ├── manifest.json
    │   ├── background.png
    │   ├── body.png
    │   ├── head.png
    │   ├── mouth_rest.png
    │   ├── mouth_ah.png
    │   ├── mouth_ee.png
    │   ├── mouth_oh.png
    │   ├── mouth_f.png
    │   ├── eye_left.png
    │   ├── eye_right.png
    │   └── pupil.png
    └── captain_bob/
        └── ... (same files)
```
