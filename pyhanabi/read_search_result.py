import os
import json
import glob
import argparse
import math
from collections import Counter

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, "seed*.json")))

    scores = []
    deviation_rates = []
    sample_rates = []
    search_position_counts: Counter = Counter()
    for path in tqdm(paths):
        with open(path) as f:
            result = json.load(f)
        scores.append(result["score"])
        if result.get("num_search_step", 0) > 0:
            deviation_rates.append(result["num_deviation"] / result["num_search_step"])
        if "success_sample_rate" in result:
            sample_rates.append(result["success_sample_rate"])
        for idx in result.get("search_indices", []):
            search_position_counts[idx] += 1

    n = len(scores)
    if n == 0:
        print(f"no seed*.json files found in {args.folder}")
        return

    mean_score = sum(scores) / n
    var_score = sum((s - mean_score) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    se_score = math.sqrt(var_score / n)

    perfects = [1.0 if s == 25 else 0.0 for s in scores]
    mean_perfect = sum(perfects) / n
    var_perfect = (
        sum((p - mean_perfect) ** 2 for p in perfects) / (n - 1) if n > 1 else 0.0
    )
    se_perfect = math.sqrt(var_perfect / n)

    print(f"folder: {args.folder}")
    print(f"num finished games: {n}")
    print(f"avg score:        {mean_score:.4f} +/- {se_score:.4f}")
    print(f"avg perfect rate: {mean_perfect * 100:.2f}% +/- {se_perfect * 100:.2f}%")
    if deviation_rates:
        print(
            f"avg deviation rate:    {sum(deviation_rates) / len(deviation_rates) * 100:.2f}%"
        )
    if sample_rates:
        print(
            f"avg success sample rate: {sum(sample_rates) / len(sample_rates) * 100:.2f}%"
        )
    if search_position_counts:
        print("search position counts:")
        for idx in sorted(search_position_counts):
            print(f"  player {idx}: {search_position_counts[idx]}")


if __name__ == "__main__":
    main()
