#!/usr/bin/env bash
set -euo pipefail

C1_EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C1_ARCH="${ARCH:-103a}"
C1_JOBS="${1:-256}"
C1_ITERS="${2:-100}"
C1_WARMUP="${3:-20}"
C1_SEED="${4:-20260910}"
C1_STAMP="$(date +%Y%m%d_%H%M%S)"

cd "$C1_EXPERIMENT_DIR"
make -B ARCH="$C1_ARCH"
mkdir -p results
C1_RESULT_FILE="results/instruction_only_${C1_ARCH}_${C1_STAMP}.txt"
./bin/instruction_only_mma "$C1_JOBS" "$C1_ITERS" "$C1_WARMUP" "$C1_SEED" \
    | tee "$C1_RESULT_FILE"
echo "wrote $C1_RESULT_FILE"

cuobjdump --dump-sass bin/instruction_only_mma > bin/instruction_only_mma.sass
echo "wrote bin/instruction_only_mma.sass"
grep -E "HMMA|TCGEN05|tcgen05" bin/instruction_only_mma.sass | head -n 40 || true
