import time
import os
import sys
import argparse
import pprint
import numpy as np
import torch

import infra  # type: ignore
import hanalearn  # type: ignore
from pyhanabi.create import create_envs, create_threads, SelfplayActGroup
from pyhanabi.eval import evaluate
import pyhanabi.common_utils as common_utils
import pyhanabi.utils as utils
import pyhanabi.rl


def clone_state_dict(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def update_ema_state(ema_state, module, decay):
    for k, v in module.state_dict().items():
        if torch.is_floating_point(v):
            ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
        else:
            ema_state[k].copy_(v)


def parse_args():
    parser = argparse.ArgumentParser(description="train RL on hanabi")
    parser.add_argument("--config", type=str, default=None)

    # training setup related
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--save_per", type=int, default=50)
    parser.add_argument("--load_model", type=str, default="None")
    parser.add_argument("--seed", type=str, default="a")
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--act_device", type=str, default="cuda:0")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # algo setting
    parser.add_argument("--ppo_clip", type=float, default=0.05)
    parser.add_argument("--ent_weight", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0, help="gae lambda")
    parser.add_argument("--multi_step", type=int, default=1)
    parser.add_argument("--shuffle_color", type=int, default=0)

    # optim setting
    parser.add_argument(
        "--lr_schedule", type=str, default=None, help="DDZ/linear/power"
    )
    parser.add_argument("--ent_schedule", type=str, default=None, help="DDZ/slow")
    parser.add_argument("--ent_decay_alpha", type=float, default=0.5)
    parser.add_argument("--ent_start_scale", type=float, default=1.0)
    parser.add_argument("--ent_end_scale", type=float, default=1.0)
    parser.add_argument("--ent_decay_power", type=float, default=2.0)
    parser.add_argument("--lr_decay_power", type=float, default=1.0)
    parser.add_argument("--schedule_update_offset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--use_old_v", type=int, default=0)
    parser.add_argument("--value_weight", type=float, default=1.0)
    parser.add_argument("--normalize_advantage", type=int, default=0)
    parser.add_argument("--mask_nonactive_step", type=int, default=0)
    parser.add_argument("--orth_init", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--batchsize", type=int, default=128)

    # model setting
    parser.add_argument(
        "--net", type=str, default="publ-lstm", help="publ-lstm/lstm/ffwd"
    )
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    # replay/data settings
    parser.add_argument("--replay_buffer_size", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")
    parser.add_argument("--eval_num_game", type=int, default=10000)
    parser.add_argument("--eval_num_thread", type=int, default=10)

    # wandb
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_project_name", type=str, default="hanabi-rl")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group_name", type=str, default=None)

    # game config
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--num_color", type=int, default=5)
    parser.add_argument("--num_rank", type=int, default=5)
    parser.add_argument("--num_hint", type=int, default=8)
    parser.add_argument("--bomb", type=int, default=0)

    args = parser.parse_args()
    args = common_utils.maybe_load_config(args)

    args.seed = utils.get_seed(args.seed)

    assert args.replay_buffer_size <= 1024
    assert args.lr_schedule in (
        None,
        "None",
        "DDZ",
        "linear",
        "power",
    ), args.lr_schedule
    assert args.ent_schedule in (None, "None", "DDZ", "slow"), args.ent_schedule
    assert 0 <= args.ent_decay_alpha <= 1, args.ent_decay_alpha
    assert args.ent_decay_power > 0, args.ent_decay_power
    assert args.lr_decay_power > 0, args.lr_decay_power
    assert args.schedule_update_offset >= 0, args.schedule_update_offset
    assert 0 <= args.ema_decay < 1, args.ema_decay
    return args


def train(args):
    common_utils.set_all_seeds(args.seed)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path, print_to_stdout=True)
    pprint.pprint(vars(args), sort_dicts=False)

    train_device = args.train_device
    act_device = args.act_device

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.bomb,
        args.max_len,
        num_color=args.num_color,
        num_rank=args.num_rank,
        num_hint=args.num_hint,
    )

    assert args.multi_step == 1
    agent = pyhanabi.rl.RLAgent(
        ppo_clip=args.ppo_clip,
        multi_step=1,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=train_device,
        net_type=args.net,
        in_dim=games[0].feature_size(False),
        hid_dim=args.rnn_hid_dim,
        out_dim=games[0].num_action(),
        num_lstm_layer=args.num_lstm_layer,
    )
    print(agent)

    if args.orth_init:
        utils.apply_orthogonal_init(agent)
        utils.verify_orth_init(agent)

    if args.load_model and args.load_model != "None":
        print("*****loading pretrained model*****")
        print(args.load_model)
        online_net = utils.get_agent_online_network(agent)
        utils.load_weight(online_net, args.load_model, train_device)
        print("***************done***************")

    saver = common_utils.TopkSaver(args.save_dir, 5)
    online_net = utils.get_agent_online_network(agent)
    optim = torch.optim.Adam(online_net.parameters(), lr=args.lr, eps=args.eps)

    ema_state = None
    ema_saver = None
    if args.ema_decay > 0:
        ema_state = clone_state_dict(online_net)
        ema_saver = common_utils.TopkSaver(os.path.join(args.save_dir, "ema"), 5)
        print(f"EMA enabled with decay={args.ema_decay}")

    replay_buffer = infra.RNNReplay(  # type: ignore
        args.replay_buffer_size,
        args.seed,
        args.prefetch,
    )

    # making input arguments
    act_group_args = {
        "devices": act_device,
        "agent": agent,
        "seed": args.seed,
        "num_thread": args.num_thread,
        "num_game_per_thread": args.num_game_per_thread,
        "num_player": args.num_player,
        "replay_buffer": replay_buffer,
        "method_batchsize": {"act": 5000},
    }

    act_group_args["actor_args"] = {
        "seed": args.seed,
        "num_player": args.num_player,
        "vdn": False,  # args.method == "vdn",
        "sad": False,  # args.sad,
        "shuffle_color": args.shuffle_color,
        "hide_action": False,
        "trinary": hanalearn.AuxType.Null,
        "multi_step": args.multi_step,
        "seq_len": args.max_len,
        "gamma": args.gamma,
    }

    act_group = SelfplayActGroup(**act_group_args)
    context, threads = create_threads(
        args.num_thread,
        args.num_game_per_thread,
        act_group.actors,
        games,
    )

    act_group.start()
    context.start()
    while replay_buffer.size() < args.replay_buffer_size:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)
    print("Success, Done")
    print("=" * 100)

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(
        args.save_dir,
        use_wandb=bool(args.use_wandb),
        wandb_exp_name=args.wandb_project_name,
        wandb_run_name=args.wandb_run_name,
        wandb_group_name=args.wandb_group_name,
        config=vars(args),
    )
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    sleep_time = 0

    for epoch in range(args.num_epoch):
        print(f"EPOCH: {epoch}")
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()

        for batch_idx in range(args.epoch_len):
            with stopwatch.time("sync"):
                num_update = (
                    args.schedule_update_offset + batch_idx + epoch * args.epoch_len
                )
                if num_update % args.actor_sync_freq == 0:
                    act_group.update_model(agent)
                torch.cuda.synchronize()

            with stopwatch.time("sample"):
                batch = replay_buffer.sample(args.batchsize, train_device)

            with stopwatch.time("forward & backward"):
                if args.ent_schedule == "DDZ":
                    k = args.num_epoch * args.epoch_len / 4
                    ent_weight = (
                        (1 + args.ent_decay_alpha)
                        * args.ent_weight
                        * min(k / max(num_update, 1), 1.0) ** args.ent_decay_alpha
                    )
                elif args.ent_schedule == "slow":
                    total_update = args.num_epoch * args.epoch_len
                    progress = min(num_update, total_update) / total_update
                    decay = progress**args.ent_decay_power
                    scale = args.ent_start_scale + decay * (
                        args.ent_end_scale - args.ent_start_scale
                    )
                    ent_weight = args.ent_weight * scale
                else:
                    ent_weight = args.ent_weight
                stat["optim/ent_weight"].append(ent_weight)

                loss = agent.loss(
                    batch,
                    ent_weight,
                    stat,
                    bool(args.use_old_v),
                    args.value_weight,
                    bool(args.normalize_advantage),
                    bool(args.mask_nonactive_step),
                )
                loss.backward()
                torch.cuda.synchronize()

            with stopwatch.time("optim step"):
                if args.lr_schedule == "DDZ":
                    # base_lr * min(k / training_iter, 1), base_lr = 2 * args.lr, k = total_steps / 4
                    k = args.num_epoch * args.epoch_len / 4
                    new_lr = 2 * args.lr * min(k / max(num_update, 1), 1.0)
                elif args.lr_schedule == "linear":
                    total_update = args.num_epoch * args.epoch_len
                    frac = 1.0 - min(num_update, total_update) / total_update
                    new_lr = args.lr * frac
                elif args.lr_schedule == "power":
                    total_update = args.num_epoch * args.epoch_len
                    progress = min(num_update, total_update) / total_update
                    frac = 1.0 - progress**args.lr_decay_power
                    new_lr = args.lr * frac
                else:
                    new_lr = args.lr

                if args.lr_schedule in ("DDZ", "linear", "power"):
                    for param_group in optim.param_groups:
                        param_group["lr"] = new_lr
                stat["optim/lr"].append(new_lr)

                g_norm = torch.nn.utils.clip_grad_norm_(
                    online_net.parameters(), args.grad_clip
                )
                optim.step()
                if ema_state is not None:
                    update_ema_state(ema_state, online_net, args.ema_decay)
                optim.zero_grad()
                torch.cuda.synchronize()

            # with stopwatch.time("sleep"):
            #     if sleep_time > 0:
            #         time.sleep(sleep_time)

            # stat["train/loss"].append(loss.detach().item())
            stat["optim/grad_norm"].append(g_norm)

        with stopwatch.time("eval & others"):
            count_factor = 1
            new_sleep_time, train_gen_ratio = tachometer.lap(
                replay_buffer,
                args.epoch_len * args.batchsize,
                count_factor,
                num_batch=args.epoch_len,
                target_ratio=None,  # args.target_data_ratio,
                current_sleep_time=sleep_time,
            )
            sleep_time = 0.6 * sleep_time + 0.4 * new_sleep_time
            stat["data/train_gen_ratio"].append(train_gen_ratio)
            print(
                f"Sleep info: new_sleep_time: {int(1000 * new_sleep_time)} MS, "
                f"actual_sleep_time: {int(1000 * sleep_time)} MS"
            )

            context.pause()
            agent.train(False)
            eval_seed = np.random.randint(100000)
            score, perfect, *_ = evaluate(
                [agent],
                args.eval_num_game,
                eval_seed,
                args.bomb,
                num_player=args.num_player,
                num_thread=args.eval_num_thread,
                num_color=args.num_color,
                num_rank=args.num_rank,
                num_hint=args.num_hint,
            )
            ema_score = None
            ema_perfect = None
            ema_model_saved = None
            if ema_state is not None:
                online_state = clone_state_dict(online_net)
                online_net.load_state_dict(ema_state)
                ema_score, ema_perfect, *_ = evaluate(
                    [agent],
                    args.eval_num_game,
                    eval_seed,
                    args.bomb,
                    num_player=args.num_player,
                    num_thread=args.eval_num_thread,
                    num_color=args.num_color,
                    num_rank=args.num_rank,
                    num_hint=args.num_hint,
                )
                online_net.load_state_dict(online_state)
            agent.train(True)
            stat["eval/score"].append(score)
            stat["eval/perfect"].append(perfect)
            force_save = (
                f"epoch{epoch + 1}" if (epoch + 1) % args.save_per == 0 else None
            )
            model_saved = saver.save(
                online_net.state_dict(),
                score,
                force_save_name=force_save,
                config=vars(args),
            )
            if ema_score is not None:
                assert ema_perfect is not None
                assert ema_saver is not None
                stat["eval/ema_score"].append(ema_score)
                stat["eval/ema_perfect"].append(ema_perfect)
                ema_model_saved = ema_saver.save(
                    ema_state,
                    ema_score,
                    force_save_name=force_save,
                    config=vars(args),
                )
            print(
                f"epoch {epoch}, "
                f"eval score: {score:.4f}, "
                f"perfect: {perfect * 100:.2f}, "
                f"model saved: {model_saved}"
            )
            if ema_score is not None:
                print(
                    f"epoch {epoch}, "
                    f"ema eval score: {ema_score:.4f}, "
                    f"ema perfect: {ema_perfect * 100:.2f}, "
                    f"ema model saved: {ema_model_saved}"
                )
            context.resume()

        stat.summary(epoch)
        stopwatch.summary()
        print("=" * 100)

    # force quit, "nicely"
    os._exit(0)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # type: ignore
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)
