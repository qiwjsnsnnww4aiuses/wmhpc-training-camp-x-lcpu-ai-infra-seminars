#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ROOT}/results/sass"
cd "${ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FLASH_KDA_CUDA_ARCHS="${FLASH_KDA_CUDA_ARCHS:-103a}"

python -m pip install -v -e . --no-build-isolation \
  2>&1 | tee results/build.log
python tests/test_reduce.py --T 1024 --H 2 --G 8 \
  2>&1 | tee results/correctness.log
python benchmarks/bench_reduce.py \
  --T "${T:-8192}" --H "${H:-96}" --G "${G:-32}" \
  --warmup "${WARMUP:-5}" --iters "${ITERS:-20}" \
  2>&1 | tee results/benchmark.log

SO_PATH="$(python -c 'import torch, flash_kda_r_C as m; print(m.__file__)')"
echo "${SO_PATH}" | tee results/sass/extension_path.txt
cuobjdump --dump-sass "${SO_PATH}" > results/sass/full.sass
grep -E 'Function :|HMMA|UTCHMMA' results/sass/full.sass \
  > results/sass/key_instructions.txt || true

echo "R standalone validation complete"
