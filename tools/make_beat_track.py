#!/usr/bin/env python
"""BotterDancer — synthetischer, rechtefreier Beat-Track mit exaktem Beat-Grid.

Erster Baustein der Audio-Lane (Produktentscheidung 04.07.: Exporte bekommen
IMMER ein eigenes/rechtefreies Audiofile — Original-Songs verlassen die
Maschine nie). Fuer echte Songs kommt spaeter Beat This! als Detektor; bei
diesem Synth ist das Beat-Grid konstruktionsbedingt exakt bekannt und wird
als JSON-Sidecar neben die WAV geschrieben (Grundlage fuer beat_cut.py und
den spaeteren Choreographer).

  python make_beat_track.py --bpm 128 --duration 6 --out beat.wav

Erzeugt: 44.1 kHz/16-bit-Stereo-WAV + <out>.beats.json
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100


def env_exp(n, decay):
    return np.exp(-np.arange(n) / (SR * decay))


def kick(dur=0.16):
    n = int(SR * dur)
    t = np.arange(n) / SR
    freq = 140.0 * np.exp(-t * 22.0) + 45.0  # Pitch-Sweep 185->45 Hz
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * env_exp(n, 0.055)
    click = np.random.default_rng(7).standard_normal(n) * env_exp(n, 0.004) * 0.35
    return (body + click) * 0.95


def snare(dur=0.14, seed=11):
    n = int(SR * dur)
    noise = np.random.default_rng(seed).standard_normal(n)
    noise = np.diff(noise, prepend=0.0)  # Hochpass-artig
    tone = np.sin(2 * np.pi * 190.0 * np.arange(n) / SR)
    return (0.7 * noise * env_exp(n, 0.030) + 0.4 * tone * env_exp(n, 0.045)) * 0.8


def hat(dur=0.05, seed=23):
    n = int(SR * dur)
    noise = np.random.default_rng(seed).standard_normal(n)
    for _ in range(2):  # doppelte Differenz = noch hoehenlastiger
        noise = np.diff(noise, prepend=0.0)
    return noise * env_exp(n, 0.012) * 0.30


def bass_note(freq, dur):
    n = int(SR * dur)
    t = np.arange(n) / SR
    saw = 2.0 * ((t * freq) % 1.0) - 1.0
    sub = np.sin(2 * np.pi * freq * 0.5 * t)
    attack = np.minimum(np.arange(n) / (SR * 0.004), 1.0)
    rel = np.ones(n)
    rel_n = int(SR * 0.02)
    rel[-rel_n:] = np.linspace(1.0, 0.0, rel_n)
    return (0.55 * saw + 0.45 * sub) * attack * rel * 0.5


def add(buf, start_s, sig):
    i = int(start_s * SR)
    j = min(i + len(sig), len(buf))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def main():
    ap = argparse.ArgumentParser(description="Rechtefreier Beat-Track mit exaktem Beat-Grid")
    ap.add_argument("--bpm", type=float, default=128.0)
    ap.add_argument("--duration", type=float, default=6.0, help="Sekunden")
    ap.add_argument("--out", required=True, help="Ziel-WAV")
    ap.add_argument("--seed", type=int, default=1, help="Variation der Bass-Figur")
    args = ap.parse_args()

    beat = 60.0 / args.bpm
    n_total = int(SR * args.duration)
    buf = np.zeros(n_total, dtype=np.float64)

    beats = []
    b = 0
    rng = np.random.default_rng(args.seed)
    # A-Moll-Figur auf Achteln: Grundton dominiert, Terz/Septime als Wuerze
    scale = [55.0, 55.0, 65.41, 55.0, 49.0, 55.0, 65.41, 73.42]
    while (t0 := b * beat) < args.duration:
        beats.append(round(t0, 6))
        add(buf, t0, kick())
        if b % 2 == 1:
            add(buf, t0, snare(seed=11 + b))
        add(buf, t0 + beat / 2.0, hat(seed=23 + b))
        for k in range(2):  # Achtel-Bass, um den Kick herum geduckt
            ts = t0 + k * beat / 2.0
            freq = scale[(2 * b + k + int(rng.integers(0, 2))) % len(scale)]
            sig = bass_note(freq, beat / 2.0 * 0.92)
            duck = np.ones(len(sig))
            duck_n = min(int(SR * 0.10), len(sig))
            if k == 0:
                duck[:duck_n] *= np.linspace(0.25, 1.0, duck_n)
            add(buf, ts, sig * duck)
        b += 1

    # Master: sanfte Saettigung + Normalisierung auf -1 dBFS
    buf = np.tanh(buf * 1.4)
    buf *= (10 ** (-1 / 20)) / max(np.abs(buf).max(), 1e-9)
    pcm = (buf * 32767).astype(np.int16)
    stereo = np.repeat(pcm[:, None], 2, axis=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())

    grid = {"bpm": args.bpm, "offset_s": 0.0, "duration_s": args.duration,
            "beats": beats, "source": "synthetic (make_beat_track.py, rechtefrei)"}
    grid_path = out.with_suffix(out.suffix + ".beats.json")
    with open(grid_path, "w") as f:
        json.dump(grid, f, indent=1)

    print(f"DONE: {out} ({args.duration:.2f}s @ {args.bpm:g} BPM, {len(beats)} Beats) + {grid_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
