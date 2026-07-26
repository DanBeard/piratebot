# PirateBot Portrait Mode — Backup Plan

## Concept

Instead of a 3D pirate standing on the porch, the prop is a **framed painting of Captain Barnacle Bill** hung on the wall. The painting is a monitor behind a custom picture frame.

When kids approach, the pirate in the painting:
- **Looks at them** (pupils track the nearest person)
- **Speaks** (audio plays, lips sync in real time from a small set of mouth shapes)
- **Reacts** (subtle head/body emotes: lean forward when curious, jolt back when surprised, tilt when grumpy)

The aesthetic is *haunted oil painting* — a little spooky, a little magical, and achievable without learning Blender.

---

## Why this wins as a backup

| Problem with 3D path | Portrait solution |
|---|---|
| Need rigged 3D model with blend shapes | 2D layers; only need a handful of PNGs |
| Pre-rendering a video per line is huge | One set of mouth shapes drives every line |
| Eye tracking needs 3D head rig | Move pupils inside eye whites with CSS |
| Adding more lines needs new assets | Just generate more WAVs; visuals stay the same |
| Godot + GPU complexity | Runs in a browser on any display PC |
| Looks like a generic tech demo | Framed monitor looks like a haunted painting |

---

## Layer architecture

The browser renders a stack of absolutely-positioned PNG layers:

```
┌─────────────────────────────┐
│  frame + matte (CSS)        │
│  ┌───────────────────────┐  │
│  │ background layer      │  │
│  │  (stormy sea, moon)   │  │
│  │                       │  │
│  │  body/base layer      │  │
│  │  (pirate head/torso)  │  │
│  │                       │  │
│  │  mouth layer          │  │
│  │  (viseme sprite)      │  │
│  │                       │  │
│  │  eye-white layer      │  │
│  │  + pupil layer        │  │
│  │    (moves via CSS)    │  │
│  │                       │  │
│  │  optional overlay:    │  │
│  │  hat, coat, lantern   │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

All layers share the same art style (oil painting / haunted portrait) so the composite looks like one image.

---

## Asset list

| Asset | Count | How to make it |
|---|---|---|
| Background | 1 | Generate with image AI: stormy sea, moonlight, ship silhouettes |
| Pirate base body | 1 | Generate portrait, remove background, cut out mouth & eyes |
| Mouth visemes | 6–8 | Inpaint or hand-tweak the mouth area into each shape |
| Eye whites | 2 (left/right) | Cut from base portrait |
| Pupils | 2 | Small dark circles; can be CSS circles or PNGs |
| Emote bodies | 3–5 optional | Generate alternate poses: laugh, surprise, grumpy |
| Frame | 1 | CSS border or a real physical frame around the monitor |

**Mouth viseme set (maps to Rhubarb output):**

| Rhubarb | Mouth shape | Description |
|---|---|---|
| `X`, `H`, `B` | `rest` | Closed / neutral |
| `A`, `D` | `aa` | Wide open "ah" |
| `C`, `E` | `ee` | Wide "ee" |
| `F`, `V` | `f` | Bottom lip under teeth |
| `G` | `oh` | Rounded "oh" |
| (optional) | `th` | Tongue visible for "th" |

---

## Asset generation pipeline

### Step 1: Base portrait

Use any image generator. Local on the 3080 with Stable Diffusion / Flux is cheapest; Midjourney/DALL-E is fastest if you already subscribe.

Prompt template:
```
Oil painting portrait of an angry skeleton pirate captain in a tricorn hat and tattered navy coat, dark stormy sea background, candle-lit, rim lighting, baroque painting style, centered, looking at viewer, mouth closed, 4k, highly detailed
```

Generate several, pick the best. The important constraints:
- Face is centered and forward-facing.
- Mouth is closed in the base image.
- Eyes are open and looking straight ahead.
- Lighting is dramatic and consistent.

### Step 2: Cut out layers

Use a free tool like **Photopea** (browser, Photoshop-like), GIMP, or Affinity Photo:

1. Remove the background → save as `body.png` with alpha.
2. Select each eye white → save as `eye_left.png`, `eye_right.png`.
3. Select the closed mouth region → save as `mouth_rest.png`.
4. If the base includes a hat/coat you want to animate separately, split those too.

### Step 3: Generate mouth visemes

Two approaches:

**A. Inpainting (best look, more iterations):**
- Mask the mouth area.
- Prompt: `open mouth ah shape, same lighting, oil painting style`.
- Generate each shape and manually align.

**B. Warping (fastest, acceptable):**
- Start from `mouth_rest.png`.
- Use OpenCV mesh warp to stretch it into open/rounded shapes.
- Add slight manual touch-up.

For a first Halloween, **warping is fine**. The effect reads at a glance.

### Step 4: Pupils

Generate or draw two small dark circles. Use CSS to position them inside the eye whites.

### Step 5: Optional emote bodies

Generate 3–5 alternate body poses. If that feels like too much work, skip it and use CSS transforms on the base layer instead:
- `scale(1.02)` for surprise
- `rotate(2deg)` for grumpy
- `translateY(-10px)` + `scale(1.01)` for laugh

---

## Runtime data flow

```
Webcam
  │
  ▼
YOLO detects person + track id
  │
  ▼
Moondream2 describes costume
  │
  ▼
PirateBot picks line from parrotts cache
  │
  ▼
Rhubarb generates visemes for that line's WAV
  │
  ▼
Browser player receives:
  - audio path
  - viseme timings
  - gaze target (person center)
  - emote hint (emotion tag)
  │
  ▼
Browser plays audio + swaps mouth sprite + tracks pupils + applies emote
```

---

## Browser player

A tiny local web app runs in fullscreen Chrome on the porch PC.

### Tech

- **HTML/CSS** for layout and painting look (CSS `border-image`, `filter`, `box-shadow`, vignette).
- **JavaScript** for WebSocket, audio playback, viseme loop, eye tracking, emotes.
- **WebSocket** connection to Python orchestrator on `ws://localhost:9877`.

### Why browser over Godot for this mode

- Easier layer compositing with CSS.
- Built-in audio + fullscreen.
- CSS filters can make it look like an actual painting instantly.
- No 3D engine overhead.

### Painting look via CSS

```css
#canvas {
  filter: contrast(1.05) sepia(0.15) saturate(0.9);
  box-shadow:
    inset 0 0 120px rgba(0,0,0,0.6),  /* vignette */
    0 0 40px rgba(0,0,0,0.8);        /* frame shadow */
}
```

Add a scanned canvas texture overlay for extra realism.

---

## Eye tracking

From YOLO, we get person bounding box center as `(x, y)` in 0–1 screen coordinates.

Browser maps that to pupil offsets:

```js
const maxOffset = 8; // pixels
const pupilX = (x - 0.5) * maxOffset * 2;
const pupilY = (y - 0.5) * maxOffset * 2;
leftPupil.style.transform = `translate(${pupilX}px, ${pupilY}px)`;
rightPupil.style.transform = `translate(${pupilX}px, ${pupilY}px)`;
```

When no person is detected, pupils drift slowly with a sine wave.

---

## Lip sync loop

Rhubarb output is a list of `{shape, start, end}` cues. The browser:

1. Starts `Audio` element.
2. In `requestAnimationFrame`, reads `audio.currentTime`.
3. Finds the current cue.
4. Sets `mouth.style.backgroundImage = url(mouth_${shape}.png)`.

If Rhubarb is missing, fallback to amplitude-based mouth opening.

---

## Emote system

Each voice line has an `emotion` tag in `voice_lines.yaml` (happy, surprised, grumpy, amused, etc.).

Browser applies a CSS class based on emotion:

| Emotion | Visual effect |
|---|---|
| `happy` | slight bounce, warm tint |
| `surprised` | quick scale up + eyes widen |
| `grumpy` | tilt, desaturate, narrowed eyes |
| `amused` | eyebrow raise (if drawn), smirk |
| `dramatic` | candle flicker overlay, red vignette pulse |

If no emote body assets exist, these are pure CSS transforms on the base layer.

---

## Scaling the corpus

This is the big win. To go from 188 to 500–2000 lines:

1. **Draft with LLM offline**: use `tools/expand_voice_lines.py` to ask the cluster LLM for 50 lines about, say, "Minecraft costumes" or "when a kid says you're fake".
2. **Curate manually**: keep only safe, funny, on-brand lines. Add an `approved: true` flag in the YAML.
3. **Batch-generate audio**: `tools/migrate_to_parrotts.py` registers the voice and submits the whole batch.
4. **Mirror to porch PC**: `tools/mirror_parrotts_cache.py` downloads all WAVs.
5. **Generate Rhubarb visemes**: `tools/batch_rhubarb.py` pre-computes viseme JSON for every WAV before showtime.

Total storage per line: ~100KB WAV + ~1KB viseme JSON. 1000 lines = ~100MB. Trivial.

---

## Safety / content guardrails

The user wants edge and specificity, but this is still a family porch.

Pipeline:
1. **LLM drafts** lines offline. No live LLM at showtime.
2. **Human approval gate** in `voice_lines.yaml`:
   ```yaml
   - id: roast_001
     text: "Oi, I'm not fake, ye ghostly muppet!"
     emotion: grumpy
     tags: [heckler, fake, comeback]
     approved: true   # only approved lines reach parrotts
   ```
3. `migrate_to_parrotts.py` only submits `approved: true` lines.
4. At showtime, PirateBot can only pick from pre-approved audio.

This gives you "it reacted to exactly what I said" moments without any risk of generating something unsafe live.

For truly unpredictable kid heckles, add fallback approved lines:
- `tags: [heckler]` → generic comebacks
- `tags: [joke]` → "Har har, ye should take that act on the high seas!"
- `tags: [fake]` → "Fake? I've been cursed for 300 years, ye landlubber!"
- `tags: [scared]` → "Don't be frightened — I only bite on Tuesdays."

---

## Deployment on the porch PC

```bash
cd ~/piratebot

# 1. One-time setup (same as 3D path)
./scripts/setup-porch.sh

# 2. Build or place portrait assets
ls portrait_viewer/assets/
# body.png, eye_left.png, eye_right.png, pupil.png,
# mouth_rest.png, mouth_aa.png, mouth_ee.png, mouth_oh.png, mouth_f.png

# 3. Pre-compute visemes for the whole corpus
uv run python tools/batch_rhubarb.py --cache-dir data/parrotts_cache

# 4. Run the portrait show
./scripts/run-portrait.sh
```

`run-portrait.sh` launches Chrome in fullscreen pointing at the local player, then starts `main.py --config config.portrait.yaml`.

---

## Physical prop

- 24–32" monitor, portrait orientation if possible.
- Monitor mounted in a thrift-store picture frame, matte cut to 16:9 if needed.
- Backlight strip behind frame for "haunted glow."
- Optional: fog machine triggered by the same detection event.

---

## Decision gates

Use this backup if any of these become true:
- No real 3D pirate model by Halloween.
- Godot 3D path proves unstable.
- You want a bigger, safer corpus than the 3D path can support.
- You want eye tracking without 3D rigging.

Switch back to 3D later if you get a rigged model; the Python orchestrator is the same.

---

## Files needed

| File | Purpose |
|---|---|
| `config.portrait.yaml` | Porch config targeting portrait avatar |
| `services/portrait_avatar.py` | WebSocket controller for the browser player |
| `portrait_viewer/index.html` | Browser player |
| `portrait_viewer/portrait.js` | Lip sync, eye tracking, emotes |
| `portrait_viewer/portrait.css` | Painting frame, layers, filters |
| `tools/batch_rhubarb.py` | Pre-compute visemes for the whole corpus |
| `tools/expand_voice_lines.py` | Draft new lines with LLM for curation |

The next step is to implement those files. The hardest creative work is generating the portrait PNGs; the code is straightforward.
