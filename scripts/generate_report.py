"""Generate the final HTML (+ optional PDF) report from ablation summaries.

    python scripts/generate_report.py \\
        --summaries outputs/ablation/summaries \\
        --output    outputs/final_report \\
        --format    both \\
        --config-snapshot configs/model_config.yaml

Reads every `<variant>.json` (produced by `scripts/evaluate.py`) from the
summaries directory, pairs each with optional CSV time series
(`<variant>_daily_returns.csv`, `<variant>_equity_curve.csv`, etc.), and
emits:

    outputs/final_report/
        ├── report.html
        ├── report.pdf        (when --format=both and WeasyPrint is installed)
        ├── plots/*.png
        └── ...

`--config-snapshot` embeds the given YAML/JSON file as a collapsible block
inside the report for reproducibility. Safe to re-run; re-emits from scratch.
"""

from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml as _yaml

from constellation_quant.outputs import ReportBuilder, load_variant_runs
from constellation_quant.utils import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries",       type=Path, default=Path("outputs/ablation/summaries"))
    p.add_argument("--output",          type=Path, default=Path("outputs/final_report"))
    p.add_argument("--template",        type=Path, default=None)
    p.add_argument("--wandb-project",   type=str,  default=None,
                   help="(Optional) pull missing summaries from a wandb project first.")
    p.add_argument("--format",          choices=["html", "both"], default="html",
                   help="'both' emits both HTML and PDF (requires weasyprint).")
    p.add_argument("--config-snapshot", type=Path, default=None,
                   help="YAML/JSON file to embed as a reproducibility snapshot.")
    return p.parse_args()


def _load_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        return _yaml.safe_load(text) or {}


def main() -> int:
    args = parse_args()
    runs = load_variant_runs(args.summaries)
    if not runs:
        log.warning("No variant summaries found in {} — emitting empty report.",
                    args.summaries)
    else:
        log.info("Loaded {} variant summaries: {}",
                  len(runs), sorted(runs.keys()))

    snapshot = _load_snapshot(args.config_snapshot)

    builder = ReportBuilder(
        runs=runs,
        out_dir=args.output,
        template_path=args.template,
        config_dump=snapshot,
    )
    out_path = builder.build(write_pdf=args.format == "both")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
