# DJI `djmd` metadata track

This format was reverse engineered from `data/berlin_3_1.MP4` (DJI NEO2,
firmware header `dvtm_NEO2.proto` 02.01.15). The decoder is
`modelforge/dji/djmd.py`.

## Container

- An MP4 data stream with codec tag `djmd`, handler "CAM meta" (stream `0:1` on
  the NEO2). It sits next to `dbgi` ("CAM dbgi", undecoded) and an MJPEG
  thumbnail.
- Each **packet is one protobuf message per video frame**, 15785 packets for
  15785 frames. Packet PTS equals frame PTS.
- There is no published schema. Wire types are standard proto3, so fields at
  their default value (0) are **omitted** and must be read as 0.

## Message layout (by field path)

| Path | Type | Meaning |
|---|---|---|
| `1.1.1` | string | schema name, `dvtm_NEO2.proto` (first packet only) |
| `1.1.5`, `1.1.10` | string | serial number, model name |
| `2.3.1/2/3` | int,int,float | video width, height, fps (first packet only) |
| `3.1.1` | int | frame counter (missing in packet 0, so 0) |
| `3.1.2` | int | device timestamp [µs], one step per frame (33366/33367) |
| **`3.2`** | | **camera record** (`3.2.1.5` = 29.97 Hz) |
| `3.2.3.1` | float | ISO |
| `3.2.4.1` | packed (num, den) | exact exposure time. The SRT shows it rounded to nominal stops (1/304 → "1/320") |
| `3.2.6.1` | int | color temperature [K] (SRT `ct`) |
| `3.2.11.1` | float | same as `abs_alt` [m] |
| `3.2.14.1` | packed (num, den) | f-number (22/10) |
| `3.2.5`, `9`, `10`, `12`, `13`, `15`, `16` | | unknown. `13` = (6400, 3609) is constant; `15.1` = 2161.78 is constant (focal length in px?); `16.1` is 4 packed uint32, varies per frame |
| **`3.4`** | | **aircraft record** (`3.4.1.5` = 10 Hz, held for 3 frames) |
| `3.4.2.1/2/3` | float | velocity **NED** [m/s], resolution 0.1. Positive D = descending |
| `3.4.3.1/2/3` | sint | pitch / roll / yaw [0.1°]. Yaw is true heading in ±180°; median offset from GPS course is 0.07° |
| `3.4.4.1.2/3` | double | latitude / longitude WGS84 [°] |
| `3.4.4.2` | int | `abs_alt` [mm] (negative on the sample, so not plain MSL) |
| `3.4.5.1` | float | `rel_alt` [mm], relative to takeoff |
| `3.4.7`, `8`, `9` | | unknown (maybe GNSS state / satellite counts) |
| **`3.5`** | | **gimbal record** (10 Hz) |
| `3.5.1.6` | int | gimbal clock [ms]. Separate clock, see Timing |
| `3.5.2.1/2/3` | sint | pitch / roll / yaw [0.1°]. Pitch is earth-referenced. Roll is always 0. The NEO2 gimbal **has no yaw axis**: yaw = aircraft yaw − 36.5° (±1°), a constant offset in the gimbal's own heading reference, so it is not the camera heading |

All position, altitude, ISO, color-temperature and f-number values match the
SRT on 15785/15785 frames.

## Other models: DJI Mavic 4 Pro

Checked on three Mavic 4 Pro clips (`dvtm_Mavic4.proto` 02.01.07, model "DJI Mavic4 Pro L3A",
Hasselblad L3D-100c, 3840×2160 @ 29.97; the longest 33 min, 59,025 packets).

- **Records are found by type, not position.** Every record of `3` starts with a header `.1`
  whose field `.1.3` is the record type: 1 camera, 2 aircraft, 3 gimbal (`.1.5` = rate in Hz,
  `.1.4` = camera name). That holds on both models; the positions differ:

  | record | NEO2 | Mavic 4 |
  |---|---|---|
  | camera | `3.2` | `3.2` |
  | aircraft | `3.4` | `3.3` |
  | gimbal | `3.5` | `3.4` |

- **Aircraft record:** the same layout as the NEO2 (velocity `.2`, attitude `.3`, lat/lon/abs_alt `.4`,
  rel_alt `.5`). `abs_alt` is about 417 m here, plausibly MSL.
- **Camera record:** ISO `.9.1` (float), exposure `.10.1` (packed num, den: 1/167…1/172),
  f-number `.11.1` (packed: 28/10 or 20/10), colour temperature probably `.24.1` (`.32.1` is a
  second, similar value; unverified). `.23.1` = 1.0 (zoom?), `.37.1` = 23–24 (focal length,
  35 mm equivalent?).
- **Gimbal record:** attitude in `.3` (pitch / roll / yaw in 0.1°; roll omitted when 0), not `.2`; plus
  a quaternion in `.4` (4 floats, unit norm). The Mavic 4 gimbal turns 360°, and its yaw is a
  **true heading**: 1.4–5.1° (median per clip) from the aircraft yaw, up to 10.6°. So the camera heading
  is the gimbal yaw (`Clip.yaw_source("auto")` decides by model: NEO → aircraft, Mavic 4 → gimbal;
  unknown models → gimbal unless it is over 20° off the aircraft yaw). The difference alone cannot
  tell: its spread (~1°) is the same as for the NEO2's constant offset.
- **Timing:** the Mavic 4 telemetry is **almost in sync**: lead 0.5 frames (17 ms; windows −0.5, 0.5,
  1.0, corr 0.87–0.98), not the NEO2's 0.7 s. There is no SRT, so it comes from optical flow.

The decoder's output on the NEO2 sample is identical before and after the type-based lookup
(15785/15785 frames).

## Other models: DJI Mavic 3

Checked on 21 clips of the user's `berlin_all` (2026-10-28; model "DJIMavic3", with SRT sidecars).

- **Records** at `3.2` / `3.3` / `3.4` (camera / aircraft / gimbal), as on the Mavic 4, but the **camera
  record has no header** `.1`: it is recognised by holding fields 9 and 10. Its fields are the Mavic 4's:
  ISO `.9.1`, exposure `.10.1` (packed 1/800), f-number `.11.1` (packed 56/10), colour temperature `.24.1`;
  `.36.1` repeats `abs_alt`, `.37.1` = 36 (focal length, 35 mm equivalent?). All agree with the SRT.
- **Latitude / longitude in radians** (`.4.1.2` / `.4.1.3`, doubles): 0.91842 = 52.6215°. The NEO2 and
  the Mavic 4 write degrees. `djmd.decode` converts (`djmd.radians` decides per clip). After that,
  djmd minus SRT is 2–3 cm median (p99 0.55 m: the SRT's 5 decimals); `rel_alt` / `abs_alt` (mm) are identical.
- **Gimbal:** attitude in `.3` and a quaternion in `.4`, as on the Mavic 4, but the yaw is in a reference
  that **shifts per flight**: −4.6 to −49° from the aircraft yaw (constant within a flight, spread ~1–3°).
  The aircraft yaw matches the GPS course within ~1° in straight flight (roll < 3°, > 4 m/s), so the camera
  heading is the **aircraft yaw** (`NO_YAW_AXIS`).

## Relation to the SRT sidecar

- The SRT `pp_current` (and `pp_target`) quaternions are (w, x, y, z), ZYX. They
  decode to (yaw = gimbal yaw, pitch = gimbal pitch, roll ≈ 0…7°), so they
  describe the camera attitude, with yaw in the same offset frame as the
  gimbal. For the true camera heading use aircraft yaw.
- `pp_current(k)` ≈ djmd gimbal at `k − 21`, median yaw error 0.07°. See Timing.

## Timing ⚠️

**djmd aircraft and gimbal records lead the video by about 0.67–0.73 s
(20–22 frames).** This was measured by correlating global optical flow
(phase correlation, 320×180) against yaw rate:

| Source | Best lag vs. image | corr |
|---|---|---|
| SRT `pp_current` yaw | 0 frames | 0.83 |
| djmd gimbal yaw | −20 frames | 0.83 |
| djmd aircraft yaw | −19 frames | 0.84 |

So image frame `k` shows the state stored in djmd packet `k − ~20`. In windows
with enough yaw motion the lead stays at 20–22 frames across the whole flight.
`gimbal_ts − frame_ts` is about 660–760 ms, in line with the lead, so the gimbal
clock is a candidate for aligning each sample. That is unverified. The aircraft
record has no clock of its own, but its yaw tracks gimbal yaw at lag 0, so it
leads by the same amount.

At 7 m/s, 0.7 s is about 5 m of position error. **Pose priors must be
time-aligned before use.**

## Open questions

- Does GPS (`3.4.4`) lead by the same 0.7 s as attitude? It's probably part of
  the same record, but this hasn't been verified against images.
- Is the exact lead best recovered from `gimbal_timestamp_ms`, or from a fixed
  per-model offset?
- Camera intrinsics: is `3.2.15` = 2161.78 a focal length in px, and for which
  resolution?
- Undecoded: the `dbgi` track (3.3 MB/s, maybe raw IMU / EIS data), and
  `3.2.16`.
