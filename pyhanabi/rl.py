import torch
import torch.nn as nn
from net import PublicLSTMPolicyNet
from common_utils import MultiCounter


class RLAgent(torch.jit.ScriptModule):
    def __init__(
        self,
        ppo_clip,
        multi_step,
        gamma,
        gae_lambda,
        device,
        net_type,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
    ):
        super().__init__()
        if net_type == "publ-lstm":
            self.policy_net = PublicLSTMPolicyNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        else:
            assert False, net_type

        self.net_type = net_type
        self.ppo_clip = ppo_clip
        self.multi_step = multi_step
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_lstm_layer = num_lstm_layer

        self.h0_keys: list[str] = list(self.policy_net.get_h0(1).keys())

    @torch.jit.script_method  # type: ignore
    def get_h0(self, batchsize: int) -> dict[str, torch.Tensor]:
        return self.policy_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        assert overwrite is None
        cloned = type(self)(
            self.ppo_clip,
            self.multi_step,
            self.gamma,
            self.gae_lambda,
            device,
            self.net_type,
            self.policy_net.in_dim,
            self.policy_net.hid_dim,
            self.policy_net.out_dim,
            self.num_lstm_layer,
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    @torch.jit.script_method  # type: ignore
    def act(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]

        # converge it hid to from batch first to batch second
        # hid size: [batch, num_layer, num_player, dim] -> [num_layer, batch x num_player, dim]
        batch, num_layer, num_player, rnn_dim = obs["h0"].size()
        hid = {}
        for k in self.h0_keys:
            hid[k] = obs[k].transpose(0, 1).flatten(1, 2).contiguous()

        action, log_pa, value, new_hid = self.policy_net.act(priv_s, legal_move, hid)

        reply = {
            "a": action.detach().cpu(),
            "log_pa": log_pa.detach().cpu(),
            "v": value.detach().cpu(),
        }

        # convert hid back to the batch first shape
        # hid size: [num_layer, batch x num_player, dim] -> [batch, num_layer, num_player, dim]
        for k, v in new_hid.items():
            v = v.transpose(0, 1).view(batch, num_layer, num_player, rnn_dim)
            reply[k] = v.detach().cpu()

        return reply

    def loss(
        self,
        batch,
        ent_weight,
        stat: MultiCounter,
        use_old_v: bool,
        value_weight: float = 1.0,
        normalize_advantage: bool = False,
        mask_nonactive_step: bool = False,
    ):
        obs = batch.obs
        hid = batch.h0
        reply = batch.action
        reward = batch.reward
        bootstrap = batch.bootstrap
        seq_len = batch.seq_len

        max_seq_len = obs["priv_s"].size(0)
        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]
        old_log_pa = reply["log_pa"]
        old_value = reply["v"]
        action = reply["a"]

        for k, v in hid.items():
            hid[k] = v.flatten(1, 2).contiguous()
            assert hid[k].sum() == 0

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        log_pa, value, ent, _ = self.policy_net(priv_s, legal_move, action, hid)

        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()

        assert self.multi_step == 1
        # episode must terminate within max_seq_len, so the last row's bootstrap
        # is always 0 — both for the actual last step (terminal) and pads beyond.
        assert (bootstrap[-1] == 0).all()

        # GAE baseline: either V_old (act-time, frozen across minibatches) or
        # the current V (recomputed each minibatch). use_old_v=True is standard
        # PPO; use_old_v=False uses the current network's value (detached).
        # δ_t = r_t + γ·V(s_{t+1})·bootstrap_t − V(s_t)
        # A_t = δ_t + γλ·bootstrap_t·A_{t+1}
        # target = A_t + V(s_t)
        with torch.no_grad():
            v_for_gae = old_value if use_old_v else value.detach()
            next_value = torch.cat(
                [v_for_gae[1:], torch.zeros_like(v_for_gae[:1])], dim=0
            )
            delta = reward + self.gamma * bootstrap * next_value - v_for_gae
            advantages = torch.zeros_like(delta)
            gae_t = torch.zeros_like(delta[0])
            for t in reversed(range(max_seq_len)):
                gae_t = delta[t] + self.gamma * self.gae_lambda * bootstrap[t] * gae_t
                # gae_t = gae_t * mask[t]
                advantages[t] = gae_t
            target_value = advantages + v_for_gae

            # print(f"seq_len[0]={seq_len[0].item()}")
            # for i in range(max_seq_len):
            #     print(
            #         f"step: {i:2d}, mask: {mask[i, 0].item(): .0f}, "
            #         f"reward: {reward[i, 0].item(): .4f}, "
            #         f"bootstrap: {bootstrap[i, 0].item(): .0f}, "
            #         f"delta: {delta[i, 0].item(): .4f}, "
            #         f"advantage: {advantages[i, 0].item(): .4f}, "
            #         f"v: {v_for_gae[i, 0].item(): .4f}, "
            #         f"target_value: {target_value[i, 0].item(): .4f}"
            #     )
            # assert False

        target_value = target_value.detach()
        adv = advantages.detach()

        value_loss = nn.functional.smooth_l1_loss(value, target_value, reduction="none")

        if normalize_advantage:
            valid_adv = adv[mask.bool()]
            adv = (adv - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)

        reweighted_pa = torch.exp(log_pa - old_log_pa)
        surr1 = reweighted_pa * adv
        surr2 = (
            torch.clamp(reweighted_pa, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv
        )
        policy_loss = -torch.min(surr1, surr2)

        assert policy_loss.size() == value_loss.size()
        assert policy_loss.size() == ent.size()

        total_num_step = mask.sum()
        assert total_num_step.item() > 0

        # off-turn steps have exactly one legal action (no-op); they carry no
        # policy/entropy gradient and dilute the average if counted. value loss
        # is unaffected — value head learns on every valid step.
        if mask_nonactive_step:
            policy_mask = (legal_move.sum(-1) > 1).float() * mask
            policy_denom = policy_mask.sum().clamp(min=1.0)
        else:
            policy_mask = mask
            policy_denom = total_num_step

        value_loss_t = (value_loss * mask).sum() / total_num_step
        policy_loss_t = (policy_loss * policy_mask).sum() / policy_denom
        ent_t = (ent * policy_mask).sum() / policy_denom

        loss = value_weight * value_loss_t + policy_loss_t - ent_weight * ent_t

        stat["train/policy_loss"].append(policy_loss_t.item())
        stat["train/value_loss"].append(value_loss_t.item())
        stat["train/ent"].append(ent_t.item())
        stat["data/game_len"].append(seq_len.mean().item())

        clipped = (surr2 < surr1).float()
        clip_frac_t = (clipped * mask).sum() / total_num_step
        stat["train/clip_frac"].append(clip_frac_t.item())

        return loss
