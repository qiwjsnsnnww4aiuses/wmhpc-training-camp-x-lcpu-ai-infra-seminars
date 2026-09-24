set -e
python -m pip install -e . --no-build-isolation
python benchmarks/bench_reduce.py "$@"
