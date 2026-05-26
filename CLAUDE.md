# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup, build, and run

The repo mixes Python (training/eval) with C++ extensions (game env, actor threading, replay buffer). The C++ side must be compiled before any Python entry point will import.

```bash
uv sync                           # install Python deps (PyTorch 2.0.1 / CUDA 11.8, pinned)
source set_env.sh                 # activates .venv, sets PYTHONPATH=$PWD:$PWD/build, OMP_NUM_THREADS=1
make                              # cmake build into ./build (produces infra.so, hanalearn.so)
```

`set_env.sh` must be sourced in every new shell — `PYTHONPATH` must include both the repo root and `build/` for `import infra` / `import hanalearn` to work.

Common entry points (run from repo root after sourcing `set_env.sh`):

```bash
python pyhanabi/train_rl.py --config configs/rl.yaml [--lr_schedule DDZ] [--seed a|b|...] --save_dir exps/...
python pyhanabi/train_belief.py --policy exps/rl/<run>/model0.pthw --save_dir exps/...
python pyhanabi/eval_run.py --folder exps/rl/<run>      # evaluates every checkpoint, writes summary.png + results.json
```

There is no test suite, no linter task, and no `make test`. Validation is by running training and watching wandb / `train.log`.

## SLURM submission

`submit.py` wraps `submitit` to launch jobs from a one-line `--cmd "..."`. Compute is selected by yaml under `configs/compute/` (e.g. `l40s.yaml`, `a5000.yaml`, `h200_4.yaml`). `--dry 1` prints without submitting; `--dry 0` actually submits.

`run.sh` is the running ledger of submitted experiments. It is a sequence of `python submit.py ... --cmd "..."` calls and is the canonical place to record/replay runs — not a script meant to be executed top-to-bottom.

## Code architecture

The training loop is a Python driver coordinating C++ self-play threads that feed a shared replay buffer.

**C++ side (`cpp/`, built via `CMakeLists.txt`):**
- `cpp/infra/` builds the `infra` pybind module: `Context` + `ThreadLoop`, `BatchRunner` (batches actor forward passes on GPU, with a model lock for hot-swapping weights), `RNNReplay` prioritized replay buffer, `Transition` / `EpisodeBuffer`.
- `cpp/rl/` builds into the `hanalearn` module: `HanabiEnv` (wraps `hanabi-learning-environment/`), `RLActor`, `HanabiThreadLoop`. Actors call into `BatchRunner` to get actions; finished episodes are pushed into the replay buffer.
- `cpp/search/` (also in `hanalearn`): top-level search (`game_sim`, `player`, `search`).
- `hanabi-learning-environment/` is a vendored submodule providing the underlying game.

**Python side (`pyhanabi/`):**
- `pyhanabi/create.py` is the glue: `create_envs`, `create_threads`, and `SelfplayActGroup` (owns one `BatchRunner` per act device and constructs the actor tree `[thread][game][player]`).
- `pyhanabi/rl.py` defines `RLAgent` (a `torch.jit.ScriptModule` so C++ actors can call its `act` method through `BatchRunner`). The policy is `PublicLSTMPolicyNet` from `pyhanabi/net.py` (`net="publ-lstm"`); it factors observation into a public part and a per-player private part, with per-player LSTM hidden state.
- `pyhanabi/train_rl.py` is the main loop: build agent → wrap in `SelfplayActGroup` → start C++ threads → warm up replay → for each epoch: `epoch_len` minibatches of `(sync model to actors, sample replay, agent.loss().backward(), optim.step())`, then `evaluate(...)` on 10k games and `TopkSaver` checkpoint. Ends with `os._exit(0)` — daemon C++ threads will not exit cleanly otherwise.
- `pyhanabi/belief_model.py` + `pyhanabi/train_belief.py` train an autoregressive belief model on top of a frozen RL policy; reuses the same actor / replay-buffer infra (with `AuxType.Full`).
- `pyhanabi/eval.py` builds a one-off `Context` of evaluation actors using `infra.BatchRunner` directly. `pyhanabi/eval_run.py` walks all `epoch*.pthw` and `model*.pthw` in a save dir, caches results to `results.json`, and emits `summary.png`.
- `pyhanabi/utils.py` has `load_agent` / `load_weight` (checkpoints carry their training config inside the `.pthw`), `Tachometer` (data/train ratio + adaptive sleep so training doesn't outrun act threads), `get_seed` (maps `"a"/"b"/...` to fixed integer seeds for reproducibility).
- `pyhanabi/common_utils/` provides `Logger` (tee stdout to `train.log`), `TopkSaver`, `MultiCounter` (metric aggregation + wandb), `Stopwatch`, `maybe_load_config` (merges yaml `--config` into argparse defaults — yaml values override argparse defaults but are overridden by explicit CLI flags).

**Schedules.** `--lr_schedule DDZ` and `--ent_schedule DDZ` are warmup-then-hold schedules computed inline in `train_rl.py` against `num_epoch * epoch_len`; both are off by default.

**`_pyhanabi/`** is an older / parallel Python tree (R2D2 main, language-belief experiments, OpenAI bot stubs). Not imported by the current RL training loop — leave alone unless explicitly working on those experiments.

## Gotchas

- Running anything Python without `source set_env.sh` will fail with `ModuleNotFoundError: infra` / `hanalearn` even after `make`.
- After editing C++, re-run `make` (it incrementally rebuilds in `build/`); restart Python — the `.so` is loaded once per process.
- `replay_buffer_size` is asserted `<= 1024` in `train_rl.py`; this is intentional (small on-policy-ish buffer), don't bump without understanding why.
- `torch==2.0.1` and the CUDA 11.8 wheel index are pinned in `pyproject.toml` because the C++ extension is built against this exact ABI.
- Checkpoints (`*.pthw`) are saved by `TopkSaver` and embed `vars(args)` as config; `utils.load_agent` reads that config back to reconstruct the agent — so don't rename agent constructor args without thinking about backward compatibility for old checkpoints.
