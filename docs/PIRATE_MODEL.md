# Pirate 3D Model Guide

The Godot project currently uses a placeholder made of CSG shapes.  For a
real Halloween show, you need a pirate character with **blend shapes** for
lip-sync.

## Required blend shapes

| Shape | Mouth pose |
|---|---|
| `viseme_aa` | Jaw open ("ah") |
| `viseme_E` | Wide ("ee") |
| `viseme_I` | Slightly open front teeth ("ih") |
| `viseme_O` | Rounded ("oh") |
| `viseme_U` | Puckered ("oo") |
| `viseme_rest` | Closed / neutral |

Optional expression shapes:

| Shape | Use |
|---|---|
| `smile` | Happy |
| `frown` | Angry / sad |
| `eyebrow_raise` | Surprised / happy |
| `eyebrow_lower` | Angry |
| `eyebrow_sad` | Sad |
| `mouth_open` | Laugh / surprise |

These names must match exactly what `godot_project/scripts/pirate_controller.gd`
references in `EXPRESSION_SHAPES` and `VISEME_SHAPES`.

## Where to get a model

### Option A: Free / CC0 sources (fast, medium quality)

1. **Sketchfab** — search "pirate rigged"
   - Filter by CC-BY or CC0
   - Download as `.glb`
   - Import into Godot: copy to `godot_project/assets/pirate_model/`

2. **RenderHub** — "Pirate" by Maksim Bugrimov (free)
   - Often FBX/MAX; import via Blender and export as `.glb`

### Option B: Mixamo + pirate clothes (medium effort, good quality)

1. Download a Mixamo character (e.g., "Maria" or "Max").
2. In Blender, add pirate clothes/hat using free CC0 assets or sculpt.
3. Add blend shapes for the visemes.
4. Export `.glb` with "Animation" and "Shape Keys" enabled.

### Option C: Buy a ready-made Godot/Unity pirate

Look for:
- Unity Asset Store pirate with blend shapes (export FBX → Blender → glb)
- TurboSquid / CGTrader "pirate cartoon rigged"

## Import into Godot

1. Copy `.glb` to `godot_project/assets/pirate_model/pirate.glb`
2. Open `godot_project/scenes/pirate.tscn`
3. Delete or hide `PlaceholderPirate`
4. Drag `pirate.glb` into the scene
5. In the imported `pirate` node:
   - Select the mesh with blend shapes
   - Assign it to `PirateAvatar.mesh` in the inspector
   - Assign the `Skeleton3D` to `PirateAvatar.skeleton`
   - Assign the `AnimationPlayer` to `PirateAvatar.animation_player`
6. Make sure the `AudioStreamPlayer3D` child stays under `PirateAvatar`

## Quick test without a model

If you don't have time to source a model, the placeholder still works — it just
won't have a real face.  Lip-sync blend shapes will be silently ignored.
For a better fallback, consider enabling the **2D talking portrait** mode
(documented separately) or just use the placeholder with expressive hat-bobbing.
