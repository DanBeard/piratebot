# PirateBot Experience Beats — Halloween 2026

This document describes the desired visitor experience, physical zones, and how software rules turn sensors + director input into prop actions.

It is a living design. As hardware is installed, update the friendly names in `props/broker/fuser_rules.yaml` and the camera/tracker config in `config.porch.yaml`.

---

## Physical layout

The house sits on a convex corner. Kids approach from the community entrance, walk along the sidewalk across the front yard, then up the driveway. To reach the candy, they must cross the side yard through a deliberately narrow entrance under the patio awning, which is draped with tarps/jute to form a dark, enclosed "spooky zone."

```
[Street] → [Sidewalk] → [Front yard / Pirate Ship] → [Driveway / Pumpkins + Cannon]
                                          ↓
                              [Side yard entrance]
                                          ↓
                    [Side yard Spooky Zone: portrait, candy, laser swamp]
```

### Zones

| Zone | ID | Location | Audio isolated? | Notes |
|------|-----|----------|-----------------|-------|
| sidewalk | `sidewalk` | Public walk | Yes | Excluded from "audience present"; used to detect arrivals only. |
| front_yard | `front_yard` | Pirate ship, fog, red/blue lights, thunder speaker | Yes | Atmospheric, automatic. |
| driveway | `driveway` | Pumpkins, cannon aimed at garage | Partially | Cannon/thunder co-exist with ship fog; pumpkins are audio-dominant. |
| graveyard | `graveyard` | Behind ship: skeleton fountain, tombstones | Mostly | Transition area; fountain pump is welcome noise. |
| sideyard | `sideyard` | Enclosed dark zone, portrait, candy, laser swamp | Yes | Portrait-only audio here; controlled choke point. |

### Cameras

| Camera | Location | Coverage | Purpose |
|--------|----------|----------|---------|
| `cam_front` | Front of house, high/wide | sidewalk, front_yard, driveway, sideyard entrance | Audience present, cannon enable, track arrivals |
| `cam_side` | Inside sideyard, looking at portrait area | sideyard | Portrait triggers, linger detection, goodbye |

Planned as PoE IP cameras delivering RTSP. Specific models TBD during remodel camera selection; choose wide-angle, decent low-light.

### Zigbee buttons

| Button | Friendly name | Single-press action |
|--------|---------------|---------------------|
| Family/spooky toggle | `btn_family_mode` | Toggle `scene.family_mode` ↔ `scene.spooky_mode`. Defaults to spooky. |
| Fire cannon | `btn_cannon` | Fire cannon once if safe (ignores audience-presence, but respects family mode). |
| Pirate/pumpkin trigger | `btn_pirate_skip` | If pumpkins singing → skip to next song; else fire a random pirate line/event. |

These are opt-in Z2M devices. They are segregated to a Halloween MQTT namespace so they do not affect year-round Home Assistant automations.

---

## Show schedule

- **Active hours:** sunset to ~11 PM / midnight.
- **Active season:** October 1–31.
- **Sunset computation:** approximate with a fixed latitude/longitude (Los Angeles area) so no internet is required.
- **Override:** director tablet can force `scene.open` or `scene.closed` regardless of sun.
- **Default scene when open but idle:** `scene.idle`.

---

## Visitor flow & beats

### Beat 1 — Approach & atmosphere (sidewalk / front yard)

**Trigger:** person detected in `front_yard` or `driveway` (not just sidewalk).

**Actions:**
- Enable low fog pulse from hidden dry-ice / smoke machine.
- Pirate ship lights red/blue; occasional random thunder + strobe.
- Parrot servo stretch goal: repeat Disney POTC parrot lines (rebuild pending).

**Audio:** ambient shanty loop from ship speaker; thunder is brief and ducking is minimal.

### Beat 2 — Driveway show (driveway)

**Trigger:** audience present in driveway.

**Actions:**
- Pumpkins auto-sing on a 10–20 minute fixed schedule.
- Cannon can auto-fire if:
  - audience present in driveway/front_yard/sideyard,
  - not in family mode,
  - pumpkins not currently singing,
  - fog machine heat cycle allows (30s–2min randomized cooldown).
- Garage ship projection plays cannonball animation when `effects.cannon.fire` is received.
- Thunder can accompany cannon.

**Audio ducking while pumpkins sing:**
- Lower ship ambient music.
- Do **not** pause portrait (sideyard is isolated).
- Cannon is muted/blocked during pumpkin songs.

### Beat 3 — Graveyard transition (graveyard)

**Trigger:** person moving from driveway toward sideyard.

**Actions:**
- Fountain pump runs (constant welcome burble).
- Optional graveyard lights / black-light flicker.
- This is mostly atmospheric; no audio ducking required.

### Beat 4 — Sideyard portrait interaction (sideyard)

**Trigger:** camera `cam_side` confirms a person in `sideyard` (fused with entrance motion if available).

**Portrait lines:**
1. **Greeting** when first detected in sideyard.
2. **Costume roast / comment** after a short look (optional VLM when available).
3. **Linger hint** at ~3 minutes: polite "arrr, pass the candy along now."
4. **Alert director** at ~5 minutes so the human pirate can intervene.
5. **Goodbye** when person leaves sideyard (track lost for >15s).

**Audio:** portrait speaker only; no ducking from front-yard effects because sideyard is isolated.

---

## Director tablet view

The control center at `http://porch-pc:9000/` should be tablet-scale and show:

- Current scene / family-vs-spooky mode.
- Zone occupancy (sidewalk, front_yard, driveway, graveyard, sideyard).
- Current audio source: ambient / pumpkins / thunder / cannon / portrait / idle.
- Manual triggers: family toggle, fire cannon, next pumpkin song, portrait test line.
- Recent events log (last N mesh messages).
- 5-minute linger alert banner.

---

## Lapel mic "agent" cues

The wireless lapel mic is routed to the porch PC. When enabled, local STT transcribes your cue and a lightweight agent maps it to a mesh action or portrait reply.

### Example cue-to-action map

| Cue | Action |
|-----|--------|
| "Tell me a joke" | Portrait plays a joke line. |
| "Who be ye?" | Portrait plays a costume-comment line (uses last VLM description if available). |
| "Fire ze cannon" / "YARR, FIRE ZE CANNON" | Fire cannon if safe. |
| "Pumpkins, sing" / "Sing, ye scurvy dogs" | Trigger next pumpkin song. |
| "Family mode" / "Gentle mode" | Toggle `scene.family_mode`. |
| "Spooky mode" | Toggle `scene.spooky_mode`. |

The mic is muted at the hardware level when not in use. STT runs locally; no cloud required.

---

## Safety / family mode

`scene.family_mode` is the spiritual successor to an E-stop. It is **not** safety-critical; it just makes the show friendlier for tiny kids.

When family mode is active:
- Master volume reduced (target ~30–50%).
- Cannon and thunder disabled.
- Portrait uses only gentle/greeting categories.
- Optional white/path lights brighter.

A big physical Zigbee button toggles it. It is also available in the director tablet and via lapel mic.

---

## Sensor fusion philosophy

1. **Camera-first**: person tracking gives the best zone and linger data.
2. **PIR optional**: Z2M PIRs can be added later as backup or to reduce camera CPU load, but fog hurts IR reliability.
3. **No beam break / pressure mat at the sideyard entrance** for now; the narrow camera view is sufficient.
4. **All Z2M devices are opt-in and Halloween-segregated** via the Z2M bridge.

---

## Fog machine protection

Cannon and any smoke-based effect enforce a 30s–2min randomized cooldown. This protects the fog machine heat cycle. The cooldown is tracked per prop in firmware, but the fuser also gates fire commands to avoid rapid re-triggering.

---

## Future / stretch goals

- Parrot servo rebuild and lines.
- More graveyard automation.
- Weather-aware line selection.
- Group-size detection.
- Actual costume VLM in portrait replies.
