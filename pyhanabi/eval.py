import numpy as np

from create import *
import infra
import utils


def evaluate(
    agents,
    num_game,
    seed,
    bomb,
    *,
    num_player=None,
    num_thread=10,
    max_len=80,
    device="cuda:0",
    num_color=5,
    num_rank=5,
    num_hint=8,
):
    """
    evaluate agents as long as they have a "act" function
    """
    if num_game < num_thread:
        num_thread = num_game

    if num_player is None:
        assert len(agents) != 1
        num_player = len(agents)

    runners = [infra.BatchRunner(agent, device, 1000, ["act"]) for agent in agents]
    context = infra.Context()
    games = create_envs(
        num_game,
        seed,
        num_player,
        bomb,
        max_len,
        num_color=num_color,
        num_rank=num_rank,
        num_hint=num_hint,
    )
    threads = []

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []
    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for player_idx in range(num_player):
                idx = player_idx % len(runners)
                actor = hanalearn.RLActor(
                    runners[idx],
                    num_player,
                    player_idx,
                    False,
                    False,
                    False,
                )

                actors.append(actor)
                all_actors.append(actor)
            thread_actors.append(actors)
            thread_games.append(games[g_idx])
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return (
        np.mean(scores),
        num_perfect / len(scores),
        scores,
        num_perfect,
        all_actors,
        games,
    )


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    device="cuda:0",
    overwrites=None,
    num_run=1,
    verbose=True,
):
    agents = []
    if overwrites is None:
        overwrites = [{} for _ in range(len(weight_files))]

    print("-" * 50)
    for i, weight_file in enumerate(weight_files):
        agent, cfg = utils.load_agent(weight_file, {"device": device})
        agents.append(agent)

        if len(overwrites[i]):
            print(f"updating cfg for {weight_file}")
            cfg.update(overwrites[i])

        agent.train(False)
    print("-" * 50)

    scores = []
    perfect = 0
    all_games = []
    all_actors = []
    for i in range(num_run):
        _, _, score, p, actors, games = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            device=device,
        )
        scores.extend(score)
        perfect += p
        all_games.extend(games)
        all_actors.extend(actors)

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print(
            "score: %.3f +/- %.3f" % (mean, sem),
            "; perfect: %.2f%%" % (100 * perfect_rate),
        )
    return mean, sem, perfect_rate, scores, all_actors, all_games
