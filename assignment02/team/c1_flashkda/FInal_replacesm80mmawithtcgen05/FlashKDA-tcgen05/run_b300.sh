#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"

ref_root=${1:-./fla_kda_ref}
result_dir=${RESULT_DIR:-../results}
mkdir -p "$result_dir" "$result_dir/env" "$result_dir/source" "$result_dir/sass"

export FLASH_KDA_CUDA_ARCHS=103a
export MAX_JOBS=${MAX_JOBS:-4}
export NVCC_THREADS=${NVCC_THREADS:-4}

hostname | tee "$result_dir/env/hostname.txt"
nvidia-smi -L | tee "$result_dir/env/nvidia_smi_L.txt"
nvcc --version | tee "$result_dir/env/nvcc_version.txt"
python --version 2>&1 | tee "$result_dir/env/python_version.txt"
python - <<'PY' | tee "$result_dir/env/torch_cuda.txt"
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name())
    print("capability:", torch.cuda.get_device_capability())
PY

grep -R -n -E 'SM80_16x8x16|tcgen05[.]mma|FLASH_KDA_TCGEN05_K2' \
  csrc/smxx setup.py > "$result_dir/source/instruction_paths.txt"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU is not visible"
cap = torch.cuda.get_device_capability()
print("device:", torch.cuda.get_device_name())
print("compute capability:", cap)
assert cap == (10, 3), f"expected B300 sm_103a, got {cap}"
PY

python -m pip install -v -e . --no-build-isolation --no-deps \
  2>&1 | tee "$result_dir/build.log"

python validation/validate.py --ref-root "$ref_root" \
  2>&1 | tee "$result_dir/correctness.log"

python validation/benchmark.py --T 8192 --H 96 --warmup 10 --iters 50 --seed 42 \
  2>&1 | tee "$result_dir/benchmark.log"

so_path=$(python - <<'PY'
import torch
import flash_kda_tcgen05_C
print(flash_kda_tcgen05_C.__file__)
PY
)
printf '%s\n' "$so_path" | tee "$result_dir/sass/extension_path.txt"
cuobjdump --dump-sass "$so_path" > "$result_dir/sass/full.sass"
awk '
  /Function :/ { function_name=$0 }
  /UTCHMMA|HMMA/ { print function_name; print $0 }
' "$result_dir/sass/full.sass" > "$result_dir/sass/key_instructions.txt"

echo
echo "ALL STEPS COMPLETED"
echo "results: $result_dir"
