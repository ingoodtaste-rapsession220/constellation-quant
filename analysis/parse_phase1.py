"""Parse Phase 1 .err logs into a CSV + per-variant summary.

Usage: python logs/parse_phase1.py
Inputs:  logs/phase1_raw/dg500-short_*.err
Outputs:
  - logs/phase1_epochs.csv  (every epoch from every run, with variant + jobid)
  - logs/phase1_summary.csv (one row per variant: best run, best epoch on each metric)
  - prints a markdown-ready summary table to stdout
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "logs" / "phase1_raw"

PAT_EPOCH = re.compile(
    r"epoch (\d+) \| train_loss=(-?\d+\.\d+) \| val_ic=(-?\d+\.\d+) \| "
    r"val_ic_ir=(-?\d+\.\d+) \| hit@50=(-?\d+\.\d+) \| spread@50=([+-]\d+\.\d+)"
)
# Identify variant from saved-checkpoint filename
PAT_VARIANT = re.compile(r"([A-I])_(?:best\.pt|epoch\d+\.pt)")


def parse_file(path: Path):
    """Returns (variant, list-of-epoch-dicts, mtime). variant=None if unparseable."""
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


JOBID_PHASE2_START = 8690000  # Phase 2 LR sweep started at jobid 8690955


def main():
    runs_by_variant = defaultdict(list)  # variant -> [{jobid, mtime, epochs}]
    for ef in sorted(RAW.glob("dg500-short_*.err")):
        jobid = int(re.search(r"_(\d+)\.err", ef.name).group(1))
        if jobid >= JOBID_PHASE2_START:
            continue  # exclude Phase 2 LR sweep jobs from Phase 1 analysis
        variant, epochs, mtime = parse_file(ef)
        if not variant or not epochs:
            continue
        runs_by_variant[variant].append(
            {"jobid": str(jobid), "mtime": mtime, "epochs": epochs}
        )

    # Pick best run per variant = run with highest peak val_ic
    best_run = {}
    for v, runs in runs_by_variant.items():
        best = max(runs, key=lambda r: max(e["val_ic"] for e in r["epochs"]))
        best_run[v] = best

    # ---------- write phase1_epochs.csv (best run per variant, all epochs) ----------
    out_epochs = ROOT / "analysis" / "phase1_epochs.csv"
    with out_epochs.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["variant", "jobid", "epoch", "train_loss",
             "val_ic", "val_ic_ir", "hit@50", "spread@50"]
        )
        for v in sorted(best_run):
            r = best_run[v]
            for e in r["epochs"]:
                w.writerow(
                    [v, r["jobid"], e["epoch"], f"{e['train_loss']:.6f}",
                     f"{e['val_ic']:.6f}", f"{e['val_ic_ir']:.4f}",
                     f"{e['hit@50']:.4f}", f"{e['spread@50']:+.6f}"]
                )

    # ---------- write phase1_summary.csv (one row per variant) ----------
    out_summary = ROOT / "analysis" / "phase1_summary.csv"
    with out_summary.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["variant", "jobid", "n_epochs",
             "peak_val_ic", "peak_val_ic_epoch",
             "peak_val_ic_ir", "peak_val_ic_ir_epoch",
             "peak_hit50", "peak_hit50_epoch",
             "peak_spread50", "peak_spread50_epoch"]
        )
        for v in sorted(best_run):
            r = best_run[v]
            eps = r["epochs"]
            def best_on(key):
                e = max(eps, key=lambda e: e[key])
                return e[key], e["epoch"]
            pv, pv_e = best_on("val_ic")
            pi, pi_e = best_on("val_ic_ir")
            ph, ph_e = best_on("hit@50")
            ps, ps_e = best_on("spread@50")
            w.writerow(
                [v, r["jobid"], len(eps),
                 f"{pv:.5f}", pv_e,
                 f"{pi:.4f}", pi_e,
                 f"{ph:.4f}", ph_e,
                 f"{ps:+.5f}", ps_e]
            )

    # ---------- print readable summary ----------
    print(f"# Phase 1 Ablation — best run per variant (parsed from {len(list(RAW.glob('*.err')))} log files)\n")
    print("| Variant | Job ID | Epochs | Peak val_ic | (ep) | Peak IR | (ep) | Peak hit@50 | (ep) | Peak spread@50 | (ep) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for v in sorted(best_run):
        r = best_run[v]
        eps = r["epochs"]
        def best_on(key):
            e = max(eps, key=lambda e: e[key])
            return e[key], e["epoch"]
        pv, pv_e = best_on("val_ic")
        pi, pi_e = best_on("val_ic_ir")
        ph, ph_e = best_on("hit@50")
        ps, ps_e = best_on("spread@50")
        rows.append((v, r["jobid"], len(eps), pv, pv_e, pi, pi_e, ph, ph_e, ps, ps_e))
    # sort by peak_val_ic desc
    for v, jid, n, pv, pv_e, pi, pi_e, ph, ph_e, ps, ps_e in sorted(rows, key=lambda x: -x[3]):
        print(f"| {v} | {jid} | {n} | **{pv:.4f}** | {pv_e} | {pi:.3f} | {pi_e} | {ph:.3f} | {ph_e} | **{ps:+.5f}** | {ps_e} |")

    print(f"\nWrote: {out_epochs}")
    print(f"Wrote: {out_summary}")
    print(f"\nVariants found: {sorted(best_run)}  (count={len(best_run)})")
    print(f"Total runs analysed: {sum(len(v) for v in runs_by_variant.values())}")


if __name__ == "__main__":
    main()
