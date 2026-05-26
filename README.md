# Ataraxos-Hanabi

GPU-accelerated self-play reinforcement learning, belief modeling, and test-time search for Hanabi.

## Requirements

- Linux
- GPU supporting CUDA 11.8
- Python 3.11 (managed by `uv`)
- PyTorch 2.0.1 (pinned in `pyproject.toml`)

## Installation

```bash
git clone --recursive git@github.com:hengyuan-hu/neo-hanabi.git
cd neo-hanabi
uv sync
source set_env.sh
make
```

`source set_env.sh` must be run once in every new shell so that the compiled
C++ modules (`infra`, `hanalearn`) are importable. A typical installation
takes a few minutes.

## Demo

Train a 2-player RL policy with the default configuration:

```bash
python pyhanabi/train_rl.py --config configs/rl.yaml
```

Logs training statistics to stdout and `train.log`.

## Usage

Train an RL policy (self-play):

```bash
python pyhanabi/train_rl.py --config configs/rl.yaml
```

Other configs in `configs/`: `rl3.yaml` (3 players), `rl4.yaml` (4 players,
hand size 4), `rl5.yaml` (5 players).

Train a belief model on top of a frozen RL policy:

```bash
python pyhanabi/train_belief.py \
    --policy exps/rl/<run>/model0.pthw \
    --save_dir exps/belief/<run>
```

Evaluate every checkpoint in a training folder:

```bash
python pyhanabi/eval_run.py --folder exps/rl/<run>
```

Run test-time search on a trained policy and belief:

```bash
python pyhanabi/search.py \
    --policy POLICY_PATH \
    --belief BELIEF_PATH \
    --search_players 1
```

## License

MIT. See [LICENSE](LICENSE).
