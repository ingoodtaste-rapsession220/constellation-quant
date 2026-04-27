"""Parse Phase 2 LR-sweep .err logs.

Stage timing on the cluster (BST):
  lr=3e-4: 20:16 → 23:11 (Apr 26)
  lr=1e-3: 00:10 → 01:08 (Apr 27)
  lr=3e-3: 02:07 → 03:05 (Apr 27)
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "logs" / "phase1_raw"  # tarball was named phase1 but has both phases

PAT_EPOCH = re.compile(
    r"epoch (\d+) \| train_loss=(-?\d+\.\d+) \| val_ic=(-?\d+\.\d+) \| "
    r"val_ic_ir=(-?\d+\.\d+) \| hit@50=(-?\d+\.\d+) \| spread@50=([+-]\d+\.\d+)"
)
PAT_VARIANT = re.compile(r"([ICD])_(?:best\.pt|epoch\d+\.pt)")

JOBID_PHASE2_START = 8690000

# Stage windows (UTC seconds since epoch — derived from mtime)
STAGES = [
    ("3em4", dt.datetime(2026, 4, 26, 20, 0), dt.datetime(2026, 4, 26, 23, 30)),
    ("1em3", dt.datetime(2026, 4, 26, 23, 30), dt.datetime(2026, 4, 27, 1, 30)),
    ("3em3", dt.datetime(2026, 4, 27, 1, 30), dt.datetime(2026, 4, 27, 3, 30)),
]


def stage_for(mtime: dt.datetime) -> str | None:
    for s, lo, hi in STAGES:
        if lo <= mtime <= hi:
            return s
    return None


def parse_file(path: Path):
    text = path.read_text(errors="replace")
    m = PAT_VARIANT.search(text)
    variant = m.group(1) if m else None
    epochs = []
    for m in PAT_EPOCH.finditer(text):
        ep, tr, vic, ir, hit, sp = m.groups()
        epochs.append(
            {
                "epoch": int(ep),
                "train_loss": float(tr),
                "val_ic": float(vic),
                "val_ic_ir": float(ir),
                "hit@50": float(hit),
                "spread@50": float(sp),
            }
        )
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    return variant, epochs, mtime


def main():
    # combo (variant, lr_stage) -> aggregated list of all epochs across chained jobs
    combos = defaultdict(list)
    for ef in sorted(RAW.glob("dg500-short_*.err")):
        jobid = int(re.search(r"_(\d+)\.err", ef.name).group(1))
        if jobid < JOBID_PHASE2_START:
            continue
        variant, epochs, mtime = parse_file(ef)
        if not variant or not epochs:
            continue
        stage = stage_for(mtime)
        if stage is None:
            continue
        for e in epochs:
            combos[(variant, stage)].append({**e, "jobid": str(jobid)})

    # Sort each combo by epoch
    for k in combos:
        combos[k].sort(key=lambda e: e["epoch"])

    # ---------- write phase2_epochs.csv ----------
    out_epochs = ROOT / "analysis" / "phase2_epochs.csv"
    with out_epochs.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["variant", "lr_stage", "jobid", "epoch", "train_loss",
             "val_ic", "val_ic_ir", "hit@50", "spread@50"]
        )
        for (v, s) in sorted(combos.keys()):
            for e in combos[(v, s)]:
                w.writerow(
                    [v, s, e["jobid"], e["epoch"],
                     f"{e['train_loss']:.6f}", f"{e['val_ic']:.6f}",
                     f"{e['val_ic_ir']:.4f}", f"{e['hit@50']:.4f}",
                     f"{e['spread@50']:+.6f}"]
                )

    # ---------- write phase2_summary.csv (one row per combo) ----------
    out_summary = ROOT / "analysis" / "phase2_summary.csv"
    with out_summary.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["variant", "lr_stage", "n_epochs",
             "peak_val_ic", "peak_val_ic_epoch",
             "peak_val_ic_ir", "peak_val_ic_ir_epoch",
             "peak_hit50", "peak_hit50_epoch",
             "peak_spread50", "peak_spread50_epoch"]
        )
        for (v, s), eps in sorted(combos.items()):
            def best_on(key):
                e = max(eps, key=lambda e: e[key])
                return e[key], e["epoch"]
            pv, pv_e = best_on("val_ic")
            pi, pi_e = best_on("val_ic_ir")
            ph, ph_e = best_on("hit@50")
            ps, ps_e = best_on("spread@50")
            w.writerow(
                [v, s, len(eps),
                 f"{pv:.5f}", pv_e,
                 f"{pi:.4f}", pi_e,
                 f"{ph:.4f}", ph_e,
                 f"{ps:+.5f}", ps_e]
            )

    # ---------- print readable summary ----------
    print("# Phase 2 LR-sweep — peak per (variant × LR) combo\n")
    print("| Variant | LR | Epochs | Peak val_ic | (ep) | Peak IR | (ep) | Peak hit@50 | (ep) | Peak spread@50 | (ep) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for (v, s), eps in sorted(combos.items()):
        def best_on(key):
            e = max(eps, key=lambda e: e[key])
            return e[key], e["epoch"]
        pv, pv_e = best_on("val_ic")
        pi, pi_e = best_on("val_ic_ir")
        ph, ph_e = best_on("hit@50")
        ps, ps_e = best_on("spread@50")
        rows.append((v, s, len(eps), pv, pv_e, pi, pi_e, ph, ph_e, ps, ps_e))
    for v, s, n, pv, pv_e, pi, pi_e, ph, ph_e, ps, ps_e in sorted(rows, key=lambda x: -x[3]):
        print(f"| {v} | {s} | {n} | **{pv:.4f}** | {pv_e} | {pi:.3f} | {pi_e} | {ph:.3f} | {ph_e} | **{ps:+.5f}** | {ps_e} |")

    print(f"\nWrote: {out_epochs}")
    print(f"Wrote: {out_summary}")
    print(f"\nCombos found: {sorted(combos.keys())}  (count={len(combos)})")


if __name__ == "__main__":
    main()
