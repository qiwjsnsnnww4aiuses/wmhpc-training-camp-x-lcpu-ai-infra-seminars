#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src="$here/FlashKDA-tcgen05"

if [[ ! -d "$src/cutlass/include/cutlass" ]]; then
  echo "ERROR: standalone CUTLASS headers are missing at:" >&2
  echo "  $src/cutlass/include/cutlass" >&2
  exit 1
fi

if [[ ! -f "$src/fla_kda_ref/naive.py" ]]; then
  echo "ERROR: standalone fla_kda_ref is missing at:" >&2
  echo "  $src/fla_kda_ref/naive.py" >&2
  exit 1
fi
mkdir -p "$here/results"

echo "source tree: $src"
echo "CUTLASS:     $src/cutlass (vendored v4.3.2)"
echo "reference:   $src/fla_kda_ref"
echo "standalone tree ready"
