import os
import sys
import time
import json
import argparse
import pprint
from typing import cast

import torch
import numpy as np

import infra
import hanalearn
from create import create_envs
from belief_model import ARBeliefModel
from rl import RLAgent
import utils
import common_utils

__GLOBAL_BATCH_RUNNER_CACHE = {}
__GLOBAL_MODEL_CACHE = {}


def get_batch_runner(model_file, model, device, methods):
    if model_file in __GLOBAL_BATCH_RUNNER_CACHE:
        return __GLOBAL_BATCH_RUNNER_CACHE[model_file]

    batch_runner = infra.BatchRunner(model, device)
    for method, bsz in methods.items():
        batch_runner.add_method(method, bsz)
    batch_runner.start()

    __GLOBAL_BATCH_RUNNER_CACHE[model_file] = batch_runner
    return batch_runner


def get_model(model_file, model_type, device):
    assert model_type in ["policy", "belief"]
    if model_file in __GLOBAL_MODEL_CACHE:
        return __GLOBAL_MODEL_CACHE[model_file]

    if model_type == "policy":
        model, _ = utils.load_agent(model_file, {"device": device})
        model.train(False)
    else:
        model = ARBeliefModel.load(model_file, device, 5)

    __GLOBAL_MODEL_CACHE[model_file] = model
    return model


class Search:
    def __init__(
        self,
        player_idx,
        device,
        bp_file,
        belief_file,
        seed,
        search_type,
        mds_eta,
        do_search=True,
    ):
        assert search_type in ["lbs", "ataraxos"], search_type
        self.player_idx = player_idx
        self.device = device
        self.do_search = do_search
        self.search_type = search_type
        self.mds_eta = mds_eta

        self.bp_model: RLAgent = get_model(bp_file, "policy", device)
        self.bp_runner = get_batch_runner(bp_file, self.bp_model, device, {"act": 5000})

        self.belief_model: ARBeliefModel | None = None
        if belief_file is not None:
            self.belief_model = cast(
                ARBeliefModel, get_model(belief_file, "belief", device)
            )

        self.rng = np.random.default_rng(seed=seed)

        self.all_players = []
        self.bp_hid: dict[str, torch.Tensor] = {}
        self.pre_act_bp_hid: dict[str, torch.Tensor] = {}
        self.belief_hid: dict[str, torch.Tensor] = {}

        self.num_search_step = 0
        self.num_deviation = 0
        self.sample_rates: list[float] = []

    def set_all_players(self, players):
        # set so that we can get correct hidden states for our partner
        self.all_players = players

    def reset(self):
        self.bp_hid = common_utils.to_device(self.bp_model.get_h0(1), self.device)  # type: ignore
        self.pre_act_bp_hid = self.bp_hid

        if self.belief_model is not None:
            self.belief_hid = common_utils.to_device(self.belief_model.get_h0(), self.device)  # type: ignore

    def get_sim_player(self):
        bp_hid = common_utils.to_device(self.pre_act_bp_hid, "cpu", detach=True)
        player = hanalearn.SimPlayer(self.player_idx, self.bp_runner, bp_hid)
        return player

    def pre_act(self):
        self.pre_act_bp_hid = self.bp_hid

    def act(self, state, game, num_search, stopwatch):
        with stopwatch.time("search_observe"):
            obs, card_count, _ = hanalearn.search_observe(state, self.player_idx)
        # the unsqueezed dim is used as seq_dim for rnn, but batch_dim for bp
        priv_s = obs["priv_s"].to(self.device).unsqueeze(0)

        belief_o = None
        if self.belief_model is not None:
            with stopwatch.time("belief_observe"):
                self.belief_hid, belief_o = self.belief_model.observe(
                    priv_s, self.belief_hid
                )

        move_scores = None
        if self.do_search and state.cur_player() == self.player_idx:
            legal_moves = state.legal_moves(self.player_idx)
            search_per_move = num_search // len(legal_moves)
            num_sample = search_per_move * 2
            print(
                f"{num_search=}, {len(legal_moves)=}, {search_per_move=}, {num_sample=}"
            )

            assert belief_o is not None
            assert self.belief_model is not None
            with stopwatch.time("belief_sample"):
                samples = self.belief_model.sample(belief_o, num_sample)
                samples = samples.cpu().squeeze(1)

            my_hand = state.hands()[self.player_idx]
            with stopwatch.time("filter_sample"):
                filtered_samples = hanalearn.filter_sample(
                    samples, card_count, game, my_hand
                )
            print(f"filter sampled hands: {num_sample} -> {len(filtered_samples)}")
            self.sample_rates.append(len(filtered_samples) / num_sample)

            if len(filtered_samples) > search_per_move:
                filtered_samples = filtered_samples[:search_per_move]
                with stopwatch.time("parallel_search_moves"):
                    move_scores = self.search(state, filtered_samples)
            elif len(filtered_samples) < 0.5 * search_per_move:
                print("too few samples, abort search")

        with stopwatch.time("bp_act"):
            legal_move = obs["legal_move"].unsqueeze(0).to(self.device)
            action, bp_log_probs, self.bp_hid = (
                self.bp_model.policy_net.act_with_log_probs(
                    priv_s, legal_move, self.bp_hid
                )
            )
            action = action.item()

        if move_scores is None:
            return action

        action = self.select_and_log_action(
            game, action, move_scores, bp_log_probs.detach().cpu()
        )
        if state.cur_player() == self.player_idx:
            print(
                f"player {self.player_idx}, move: {game.get_move(action).to_string()}"
            )

        return action

    def search(self, state, samples):
        print("search per move:", len(samples))
        sim_seeds = self.rng.integers(low=1, high=int(1e8), size=len(samples))
        search_players = [player.get_sim_player() for player in self.all_players]

        legal_moves = state.legal_moves(self.player_idx)
        scores = []
        # for move in legal_moves:
        #     score = hanalearn.search_move(
        #         state, move, samples, sim_seeds, self.player_idx, search_players
        #     )
        #     print(move.to_string(), score)
        #     scores.append(score)

        scores = hanalearn.parallel_search_moves(
            state, legal_moves, samples, sim_seeds, self.player_idx, search_players
        )
        move_scores = list(zip(legal_moves, scores))
        return move_scores

    def select_and_log_action(self, game, bp_action, move_scores, bp_log_probs) -> int:
        best_action = -1
        best_combined = -1
        bp_combined = -1

        combined_scores = {}

        for move, score in move_scores:
            move_uid = game.get_move_uid(move)
            if self.search_type == "ataraxos":
                combined = bp_log_probs[move_uid].item() + self.mds_eta * score
            else:
                combined = score

            combined_scores[move.to_string()] = combined

            if move_uid == bp_action:
                bp_combined = combined + (0.05 if self.search_type == "lbs" else 0.0)

            if combined > best_combined:
                best_combined = combined
                best_action = move_uid

        if bp_combined < best_combined:
            action = best_action
        else:
            action = bp_action

        self.num_search_step += 1
        if action != bp_action:
            self.num_deviation += 1

        for move, score in move_scores:
            move_uid = game.get_move_uid(move)
            info = f"{move.to_string()}: q={score:.2f}, "

            combined_score = combined_scores[move.to_string()]
            if self.search_type == "ataraxos":
                info += (
                    f"log_pi={bp_log_probs[move_uid].item():.2f}, "
                    f"mds={combined_score:.2f}"
                )
            else:
                info += f"sum: {combined_score:.2f}"
            if move_uid == bp_action:
                info = f"{info}, (bp)"
            if move_uid == best_action:
                info = f"{info}, (best)"
            if move_uid == action:
                info = f"{info}, (selected)"
            print(info)

        return action


def run_game(
    seed,
    num_search,
    save_dir,
    policy,
    belief,
    search_type,
    search_players: int,
    mds_eta,
):
    result_path = os.path.join(save_dir, f"seed{seed}.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        score = result["score"]
        print(f"seed {seed}: result exists at {result_path}, skip, score: {score}")
        return score

    cfg = utils.get_train_config(policy)
    num_player = cfg["num_player"]
    assert (
        search_players <= num_player
    ), f"search_players ({search_players}) must be <= num_player ({num_player})"

    rng = np.random.default_rng(seed=seed)
    player_seeds = rng.integers(low=1, high=int(1e4), size=num_player)
    search_indices = set(
        rng.choice(num_player, size=search_players, replace=False).tolist()
    )
    print(f"search players: {sorted(search_indices)}")

    players = [
        Search(
            player_idx=i,
            device="cuda",
            bp_file=policy,
            belief_file=belief,
            seed=int(player_seeds[i]),
            do_search=(i in search_indices),
            search_type=search_type,
            mds_eta=mds_eta,
        )
        for i in range(num_player)
    ]
    for player in players:
        if player.do_search:
            player.set_all_players(players)

    game = create_envs(
        num_env=1,
        seed=seed,
        num_player=len(players),
        bomb=0,
        max_len=-1,
        random_start_player=cfg.get("random_start_player", 1),
        num_color=cfg.get("num_color", 5),
        num_rank=cfg.get("num_rank", 5),
        num_hint=cfg.get("num_hint", 8),
    )[0]
    game.reset()
    for player in players:
        player.reset()

    stopwatch = common_utils.Stopwatch()
    game_t = time.time()
    step = 0
    with torch.no_grad():
        while not game.terminated():
            print(f"===== Step {step} curr player: {game.get_current_player()} =====")
            print(game.get_hle_state().to_string())

            t = time.time()
            for player in players:
                player.pre_act()
            actions = []
            for player in players:
                actions.append(
                    player.act(
                        game.get_hle_state(),
                        game.get_hle_game(),
                        num_search,
                        stopwatch,
                    )
                )
            action = actions[game.get_current_player()]
            move = game.get_move(action)
            print(f"move: {move.to_string()}")
            game.step(move)
            step += 1
            print(f"time {time.time() - t:.1f}")

    score = game.get_score()
    print(f"game end, len: {step}, score: {score}")
    print(f"total time: {time.time() - game_t:.2f}")
    print(f"seed {seed}: {score}")
    stopwatch.summary()
    all_sample_rates = []
    for i, player in enumerate(players):
        if player.do_search:
            print(
                f"player {i}: {player.num_deviation}/{player.num_search_step} deviations"
            )
            all_sample_rates.extend(player.sample_rates)

    num_deviation = sum(p.num_deviation for p in players if p.do_search)
    num_search_step = sum(p.num_search_step for p in players if p.do_search)
    success_sample_rate = (
        sum(all_sample_rates) / len(all_sample_rates) if all_sample_rates else 0.0
    )
    print(f"success_sample_rate: {success_sample_rate:.4f}")
    result = {
        "seed": seed,
        "score": score,
        "num_deviation": num_deviation,
        "num_search_step": num_search_step,
        "success_sample_rate": success_sample_rate,
        "search_type": search_type,
        "search_players": search_players,
        "search_indices": sorted(search_indices),
        "mds_eta": mds_eta,
    }
    with open(result_path, "w") as f:
        json.dump(result, f)
    return score


NUM_JOB = 100


def submit_array(args):
    import submitit

    assert (
        args.num_game % NUM_JOB == 0
    ), f"num_game ({args.num_game}) must be divisible by NUM_JOB ({NUM_JOB})"
    games_per_job = args.num_game // NUM_JOB

    os.makedirs(args.save_dir, exist_ok=True)
    executor = submitit.AutoExecutor(folder=os.path.join(args.save_dir, "submitit"))
    executor.update_parameters(
        slurm_account="iliad",
        slurm_partition="sc-loprio",
        cpus_per_task=16,
        slurm_gres="gpu:1",
        slurm_mem="64gb",
        slurm_time=20 * games_per_job,
        slurm_job_name="search",
        slurm_constraint="ampere|ada|hopper",
    )

    jobs = []
    with executor.batch():
        for i in range(NUM_JOB):
            seed = args.seed + i * games_per_job
            cmd = [
                "python",
                "-u",
                "pyhanabi/search.py",
                "--save_dir",
                args.save_dir,
                "--num_search",
                str(args.num_search),
                "--seed",
                str(seed),
                "--num_game",
                str(games_per_job),
                "--policy",
                args.policy,
                "--belief",
                args.belief,
                "--search_type",
                args.search_type,
                "--search_players",
                str(args.search_players),
                "--mds_eta",
                str(args.mds_eta),
            ]
            print(f"job {i}: {' '.join(cmd)}")
            jobs.append(executor.submit(submitit.helpers.CommandFunction(cmd)))
    print(f"submitted {len(jobs)} jobs, ids: {[j.job_id for j in jobs[:3]]}...")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--save_dir", type=str, default="exps/search")
    parser.add_argument("--num_search", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_game", type=int, default=1)
    parser.add_argument("--submit", type=int, default=0)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--belief", type=str, required=True)
    parser.add_argument("--search_type", choices=["lbs", "ataraxos"], default="ataraxos")
    parser.add_argument("--search_players", type=int, default=1)
    parser.add_argument("--mds_eta", type=float, default=30.0)

    args = parser.parse_args()
    pprint.pprint(vars(args))

    if args.submit:
        submit_array(args)
        return

    original_stdout = sys.stdout
    for i in range(args.num_game):
        seed = args.seed + i
        log_path = os.path.join(args.save_dir, f"seed{seed}_game.txt")
        sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)
        run_game(
            seed,
            args.num_search,
            args.save_dir,
            args.policy,
            args.belief,
            args.search_type,
            args.search_players,
            args.mds_eta,
        )
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
