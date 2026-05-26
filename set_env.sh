unset PYTHONPATH
export PYTHONPATH=$PWD:$PWD/build:$PYTHONPATH
export OMP_NUM_THREADS=1

user="${USER:-$(whoami)}"
source .venv/bin/activate
