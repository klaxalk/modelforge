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
