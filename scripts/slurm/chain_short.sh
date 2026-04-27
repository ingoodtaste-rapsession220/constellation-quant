#!/bin/bash
# Submit a chain of N gpushort jobs for one variant — each auto-resumes the
# previous via --dependency=afterany. Each job trains for ~55 min, saves
# every epoch, then the next job picks up from the latest checkpoint.
#
# gpushort limits: 1 h per job, 2 jobs in queue at once. So chain 2 variants
# in parallel, each with its own chain of this script.
#
#   bash scripts/slurm/chain_short.sh I 8          # 8 jobs (~8 hours wall)
#   bash scripts/slurm/chain_short.sh A 2          # 2 jobs for baseline
#
# Arg 1: variant name
# Arg 2: number of chained jobs (default 6)

set -euo pipefail

VARIANT="${1:?usage: $0 VARIANT [N_JOBS]}"
N_JOBS="${2:-6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_short.sh"

echo "Chaining $N_JOBS × gpu-short jobs for variant $VARIANT"

PREV=""
for ((i = 1; i <= N_JOBS; i++)); do
    if [[ -z "$PREV" ]]; then
        JOB_ID=$(sbatch --parsable "$TRAIN_SCRIPT" "$VARIANT")
    else
        JOB_ID=$(sbatch --parsable --dependency=afterany:"$PREV" "$TRAIN_SCRIPT" "$VARIANT")
    fi
    echo "  job $i: $JOB_ID   (starts after $PREV)"
    PREV="$JOB_ID"
done

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel chain:  scancel -u \$USER -n cq-short"
