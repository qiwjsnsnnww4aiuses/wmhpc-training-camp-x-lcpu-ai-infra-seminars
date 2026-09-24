set -e
python -m pip install -e . --no-build-isolation
python tests/test_reduce.py "$@"
