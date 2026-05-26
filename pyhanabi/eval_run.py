"""Evaluate every checkpoint in a training folder and plot results.

Run from the project root with PYTHONPATH set (see set_env.sh):

    python pyhanabi/eval_run.py --folder exps/rl/DDZ_run2_seed1
    python pyhanabi/eval_run.py --weight exps/rl/DDZ_run2_seed1/model0.pthw
"""

import argparse
import math
import os
import re
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyhanabi.eval import evaluate_saved_model
from pyhanabi.utils import get_train_config

CKPT_RE = re.compile(r"^(epoch|model)(\d+)\.pthw$")


def collect_checkpoints(folder):
    epochs, models = [], []
    for name in os.listdir(folder):
        m = CKPT_RE.match(name)
        if m is None:
            continue
        kind, num = m.group(1), int(m.group(2))
        path = os.path.join(folder, name)
        if kind == "epoch":
            epochs.append((num, path))
        else:
            models.append((num, path))
    epochs.sort(key=lambda x: x[0])
    models.sort(key=lambda x: x[0])
    return epochs, models


def perfect_sem_from_scores(scores):
    n = len(scores)
    if n <= 1:
        return 0.0
    perfects = [1.0 if s == 25 else 0.0 for s in scores]
    mean_perfect = sum(perfects) / n
    var_perfect = sum((p - mean_perfect) ** 2 for p in perfects) / (n - 1)
    return math.sqrt(var_perfect / n)


def eval_one(weight_file, num_game, seed, bomb, device):
    cfg = get_train_config(weight_file)
    num_player = int(cfg["num_player"])
    mean, sem, perfect_rate, scores, *_ = evaluate_saved_model(
        [weight_file] * num_player,
        num_game,
        seed,
        bomb,
        device=device,
        num_run=1,
        verbose=False,
    )
    scores = [float(s) for s in scores]
    return mean, sem, perfect_rate, perfect_sem_from_scores(scores), scores


def _annotate(ax, xs, ys, fmt="{:.3f}"):
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(y),
            xy=(x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )


def plot_all(epoch_results, model_results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if epoch_results:
        xs = [r["num"] for r in epoch_results]
        scores = [r["score"] for r in epoch_results]
        sems = [r["sem"] for r in epoch_results]
        perfects = [100 * r["perfect"] for r in epoch_results]
        perfect_sems = [100 * r.get("perfect_sem", 0.0) for r in epoch_results]

        ax = axes[0, 0]
        ax.errorbar(xs, scores, yerr=sems, marker="o", color="tab:blue")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.set_title("epoch ckpts: score")
        _annotate(ax, xs, scores)

        ax = axes[0, 1]
        ax.errorbar(xs, perfects, yerr=perfect_sems, marker="s", color="tab:red")
        ax.set_xlabel("epoch")
        ax.set_ylabel("perfect rate (%)")
        ax.set_title("epoch ckpts: perfect rate")
        _annotate(ax, xs, perfects)
    else:
        axes[0, 0].set_axis_off()
        axes[0, 1].set_axis_off()

    if model_results:
        labels = [f"model{r['num']}" for r in model_results]
        xs = list(range(len(labels)))
        scores = [r["score"] for r in model_results]
        sems = [r["sem"] for r in model_results]
        perfects = [100 * r["perfect"] for r in model_results]
        perfect_sems = [100 * r.get("perfect_sem", 0.0) for r in model_results]

        ax = axes[1, 0]
        ax.bar(xs, scores, yerr=sems, color="tab:blue")
        ax.set_ylabel("score")
        ax.set_title("saved-best models: score")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        _annotate(ax, xs, scores)

        ax = axes[1, 1]
        ax.bar(xs, perfects, yerr=perfect_sems, color="tab:red")
        ax.set_ylabel("perfect rate (%)")
        ax.set_title("saved-best models: perfect rate")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        _annotate(ax, xs, perfects)
    else:
        axes[1, 0].set_axis_off()
        axes[1, 1].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--folder", help="evaluate all epoch*/model*.pthw in this folder"
    )
    group.add_argument(
        "--weight", help="evaluate a single .pthw checkpoint (results not saved)"
    )
    parser.add_argument("--num_game", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bomb", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--save_dir", default=None, help="output dir (default: <folder>/eval_run)"
    )
    args = parser.parse_args()

    if args.weight is not None:
        print(f"evaluating {args.weight}")
        mean, sem, perfect, perfect_sem, _ = eval_one(
            args.weight, args.num_game, args.seed, args.bomb, args.device
        )
        print(
            f"score: {mean:.4f} +/- {sem:.4f}, "
            f"perfect: {100 * perfect:.2f}% +/- {100 * perfect_sem:.2f}%"
        )
        return

    save_dir = args.save_dir or os.path.join(args.folder, "eval_run")
    os.makedirs(save_dir, exist_ok=True)
    cache_path = os.path.join(save_dir, "results.json")

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            blob = json.load(f)
        for entry in blob.get("epoch", []) + blob.get("model", []):
            cache[entry["path"]] = entry
        print(f"loaded {len(cache)} cached results from {cache_path}")

    epochs, models = collect_checkpoints(args.folder)
    print(f"found {len(epochs)} epoch ckpts, {len(models)} model ckpts")

    def run_group(items, tag):
        results = []
        for num, path in items:
            cached = cache.get(path)
            if cached is not None and "scores" in cached:
                if "perfect_sem" not in cached:
                    cached["perfect_sem"] = perfect_sem_from_scores(cached["scores"])
                print(
                    f"[{tag}] cached {os.path.basename(path)}: "
                    f"score {cached['score']:.4f} +/- {cached['sem']:.4f}, "
                    f"perfect {100 * cached['perfect']:.2f}% +/- {100 * cached['perfect_sem']:.2f}%"
                )
                results.append(cached)
                continue
            print(f"[{tag}] evaluating {os.path.basename(path)}")
            mean, sem, perfect, perfect_sem, scores = eval_one(
                path, args.num_game, args.seed, args.bomb, args.device
            )
            print(
                f"    score: {mean:.4f} +/- {sem:.4f}, "
                f"perfect: {100 * perfect:.2f}% +/- {100 * perfect_sem:.2f}%"
            )
            entry = {
                "num": num,
                "path": path,
                "score": float(mean),
                "sem": float(sem),
                "perfect": float(perfect),
                "perfect_sem": float(perfect_sem),
                "scores": scores,
            }
            results.append(entry)
            cache[path] = entry
            # write after every eval so progress survives crashes
            _write_cache(cache_path, results_so_far(cache, epochs, models), args)
        return results

    epoch_results = run_group(epochs, "epoch")
    model_results = run_group(models, "model")

    _write_cache(cache_path, {"epoch": epoch_results, "model": model_results}, args)

    if epoch_results or model_results:
        plot_all(epoch_results, model_results, os.path.join(save_dir, "summary.png"))

    print(f"wrote results + plots to {save_dir}")


def results_so_far(cache, epochs, models):
    epoch_results = [cache[p] for _, p in epochs if p in cache]
    model_results = [cache[p] for _, p in models if p in cache]
    return {"epoch": epoch_results, "model": model_results}


def _write_cache(path, results, args):
    with open(path, "w") as f:
        json.dump({**results, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
