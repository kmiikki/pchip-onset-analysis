#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic rgb-tr.csv demonstration data for the PCHIP onset workflow.

Version 3:
- transparent, flat RGB/BW baseline before T1
- monotonic smoothstep T1 transition, avoiding artificial undershoot
- stronger monotonic T2 transition
- slow post-T2 saturation tail

The generated data are synthetic demonstration data. They are not
experimental results and must not be used for scientific conclusions.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def smootherstep01(x: np.ndarray) -> np.ndarray:
    """Return a monotonic C2 smootherstep from 0 to 1 for x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (x * (x * 6.0 - 15.0) + 10.0)


def cooling_smootherstep(temp_c: np.ndarray, start_c: float, width_c: float) -> np.ndarray:
    """
    Smooth monotonic transition during cooling.

    start_c is the temperature where the signal starts to change.
    The transition reaches full amplitude at start_c - width_c.
    """
    x = (start_c - temp_c) / width_c
    return smootherstep01(x)


def main() -> None:
    out_csv = Path("rgb-tr.csv")

    n = 6000
    dt_s = 2.0
    start_dt = datetime(2026, 5, 29, 9, 0, 0)
    start_ts = int(start_dt.timestamp())
    rng = np.random.default_rng(20260529)

    time_s = np.arange(n, dtype=float) * dt_s
    tr = np.linspace(60.0, 25.0, n)

    t1_c = 53.7
    t2_c = 48.8

    baseline_r = 38.0
    baseline_g = 33.4
    baseline_b = 35.2

    s1 = cooling_smootherstep(tr, t1_c, 0.95)
    post_t1_tail = np.where(tr <= t1_c, 1.0 - np.exp(-(t1_c - tr) / 4.5), 0.0)

    s2 = cooling_smootherstep(tr, t2_c, 1.35)
    post_t2_tail = np.where(tr <= t2_c, 1.0 - np.exp(-(t2_c - tr) / 6.0), 0.0)
    late_tail = np.where(tr <= 42.0, 1.0 - np.exp(-(42.0 - tr) / 9.0), 0.0)

    r_clean = baseline_r + 15.0 * s1 + 7.0 * post_t1_tail + 56.0 * s2 + 31.0 * post_t2_tail + 5.0 * late_tail
    g_clean = baseline_g + 13.0 * s1 + 6.0 * post_t1_tail + 53.0 * s2 + 30.0 * post_t2_tail + 4.5 * late_tail
    b_clean = baseline_b + 11.0 * s1 + 5.0 * post_t1_tail + 45.0 * s2 + 28.0 * post_t2_tail + 4.0 * late_tail

    noise_scale = 0.70 + 0.35 * s1 + 0.35 * s2 + 0.20 * post_t2_tail
    r = np.clip(r_clean + rng.normal(0, noise_scale, n), 0, 255)
    g = np.clip(g_clean + rng.normal(0, noise_scale, n), 0, 255)
    b = np.clip(b_clean + rng.normal(0, noise_scale, n), 0, 255)
    bw = (r + g + b) / 3.0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Datetime", "Timestamp", "Time (s)", "Tr (°C)", "BW", "R", "G", "B"])
        for i in range(n):
            dt = start_dt + timedelta(seconds=float(time_s[i]))
            ts = start_ts + int(time_s[i])
            writer.writerow([
                str(i + 1),
                " " + dt.strftime("%Y-%m-%d %H:%M:%S"),
                str(ts),
                f"{time_s[i]:.1f}",
                f"{tr[i]:.10f}",
                f"{bw[i]:.10f}",
                f"{r[i]:.10f}",
                f"{g[i]:.10f}",
                f"{b[i]:.10f}",
            ])

    print(f"Wrote: {out_csv}")
    print(f"Rows: {n}")
    print(f"Interval: {dt_s:.1f} s")
    print(f"Temperature: {tr[0]:.1f} -> {tr[-1]:.1f} °C")
    print(f"Synthetic T1 start: {t1_c:.1f} °C")
    print(f"Synthetic T2 start: {t2_c:.1f} °C")


if __name__ == "__main__":
    main()
