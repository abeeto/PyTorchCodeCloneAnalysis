from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union, Tuple

import warnings
import torch
import gym
import tianshou
import numpy as np
import scipy

from torch.nn import functional as F
from scipy.optimize import minimize

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import BasePolicy
from tianshou.utils.net.continuous import Actor as ContinuousActor
from tianshou.utils.net.discrete import Actor as DiscreteActor


def _dual(eta: np.ndarray, target_q: np.ndarray, epsilon: float):
    """Dual function of the non-parametric variational

    g(eta) = eta * dual_constraint + eta sum{log(sum{exp(Q(s,a)/eta)})}
    """

    max_q = np.max(target_q, -1)
    new_eta = (
        eta * epsilon
        + np.mean(max_q)
        + eta
        * np.mean(np.log(np.mean(np.exp((target_q - max_q[:, None]) / eta), axis=1)))
    )
    return new_eta


def get_dist_fn(discrete: bool):
    if discrete:
        return torch.distributions.Categorical
    else:

        def fn(logits):
            return torch.distributions.Independent(
                torch.distributions.Normal(*logits), 1
            )

        return fn


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


class MPOPolicy(BasePolicy):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor: Optional[torch.nn.Module],
        actor_optim: Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        epsilon: float = 0.1,
        alpha: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        actor_grad_norm: float = 5.0,
        critic_grad_norm: float = 5.0,
        mstep_iter_num: int = 5,
        critic_loss_type: str = "mse",
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        estimation_step: int = 1,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs,
        )

        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim

        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau

        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma

        self._noise = exploration_noise
        self._rew_norm = reward_normalization
        self._n_step = estimation_step

        # initialize Lagrange multiplier
        self._eta = np.random.rand()
        self._eta_kl = 0.0
        self._epsilon_kl = 0.01
        self._epsilon = epsilon
        self._alpha = alpha
        self._actor_grad_norm = actor_grad_norm
        self._critic_grad_norm = critic_grad_norm
        self._discrete_act = isinstance(actor, DiscreteActor)
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._device = device
        self._norm_critic_loss = (
            torch.nn.MSELoss() if critic_loss_type == "mse" else torch.nn.SmoothL1Loss()
        )
        self._mstep_iter_num = mstep_iter_num
        self.dist_fn = get_dist_fn(self._discrete_act)

    def set_exp_noise(self, noise: Optional[BaseNoise]):
        self._noise = noise

    def train(self, mode: bool = True) -> "MPOPolicy":
        """Switch to trainining mode.

        Args:
            mode (bool, optional): Enable training mode or not.. Defaults to True.

        Returns:
            MPOPolicy: MPO policy instance.
        """

        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        target_q = self.critic_old(
            batch.obs_next, self(batch, model="actor_old", input="obs_next").act
        )
        # also save target q
        batch["target_q"] = target_q
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:

        super().post_process_fn(batch, buffer, indices)
        returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            gamma=self._gamma,
            gae_lambda=1.0,
        )

        batch["returns"] = returns
        batch["advantages"] = advantages

        batch.to_torch(device=self._device)

        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[Dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        logits, hidden = model(obs, state=state, info=batch.info)

        if not self.training:
            # arg max
            if self._discrete_act:
                actions = F.softmax(logits, dim=-1).argmax(-1)
            else:
                actions = logits[0]  # get means as actions
        else:
            actions = self.dist_fn(logits).sample()

        # clip actions
        if not self._discrete_act:
            actions = torch.clip(actions, -1.0, 1.0)
        return Batch(act=actions, state=hidden)

    def critic_update(self, batch: Batch, particle_num: int = 64):
        batch_size = batch.obs.size(0)
        with torch.no_grad():
            logits, _ = self.actor_old(batch.obs_next)
            policy: torch.distributions.Distribution = self.dist_fn(logits)
            sampled_next_actions = policy.sample((particle_num,)).transpose(
                0, 1
            )  # (batch_size, sample_num, action_dim)
            expaned_next_states = batch.obs_next[:, None, :].expand(
                -1, particle_num, -1
            )  # (batch_size, sample_num, obs_dim)

            # get expected Q vaue from target critic
            next_q_values = self.critic_old(
                expaned_next_states.reshape(-1, self._obs_dim),
                sampled_next_actions.reshape(-1, self._act_dim),
            )
            next_state_values = next_q_values.reshape(batch_size, -1).mean(-1)
            assert batch.rew.shape == next_state_values.shape, (
                batch.rew.shape,
                next_state_values.shape,
            )
            y = batch.rew.float() + self._gamma * next_state_values

        self.critic_optim.zero_grad()
        q_values = self.critic(batch.obs, batch.act).squeeze()
        loss = self._norm_critic_loss(q_values, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self._critic_grad_norm)
        self.critic_optim.step()
        return loss, y

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        batch_size = batch.obs.size(0)

        # compute critic loss
        particle_num = 64
        critic_loss, q_label = self.critic_update(batch, particle_num=particle_num)
        mean_est_q = q_label.abs().mean()

        # E-step for policy improvement
        with torch.no_grad():
            reference_logits, _ = self.actor_old(batch.obs)
            reference_policy = self.dist_fn(reference_logits)
            sampled_actions = reference_policy.sample(
                (particle_num,)
            )  # (K, batch_size, act_dim)
            expanded_states = batch.obs[None, ...].expand(
                particle_num, batch_size, -1
            )  # (K, batch_size, obs_dim)
            target_q = self.critic_old(
                expanded_states.reshape(-1, self._obs_dim),
                sampled_actions.reshape(-1, self._act_dim),
            ).reshape(
                particle_num, batch_size
            )  # (K, batch_size)
            target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (batch_size, K)
            reference_log_prob = reference_policy.expand(
                (particle_num, batch_size)
            ).log_prob(sampled_actions)

        self._eta = minimize(
            _dual,
            np.asarray([self._eta]),
            args=(target_q_np, self._epsilon),
            method="SLSQP",
            bounds=[(1e-6, None)],
        ).x[0]

        # M-step: update actor based on Lagrangian
        average_actor_loss = 0.0
        average_kl = 0.0
        # normalize q
        norm_target_q = F.softmax(target_q / self._eta, dim=0)  # (K, batch_size)
        for _ in range(self._mstep_iter_num):
            logits, _ = self.actor(batch.obs)
            policy = self.dist_fn(logits)
            log_prob = policy.expand((particle_num, batch_size)).log_prob(
                sampled_actions
            )
            mle_loss = torch.mean(norm_target_q * log_prob)  # (K, batch_size)

            # compute KL divergence
            kl_to_ref_policy = F.kl_div(log_prob, reference_log_prob)
            average_kl += kl_to_ref_policy.item() / self._mstep_iter_num

            # Update lagrange multipliers by gradient descent
            self._eta_kl -= self._alpha * (self._epsilon_kl - kl_to_ref_policy.item())
            self._eta_kl = max(self._eta_kl, 0.0)

            self.actor_optim.zero_grad()

            actor_loss = -(
                mle_loss + self._eta_kl * (self._epsilon - kl_to_ref_policy.mean())
            )
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)

            self.actor_optim.step()
            average_actor_loss += actor_loss.item()

        self.sync_weight()

        return {
            "loss/actor": average_actor_loss,
            "loss/critic": critic_loss.item(),
            "est/q": mean_est_q.item(),
            "est/kl": average_kl,
            "est/eta": self._eta,
        }

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            act = act + self._noise(act.shape)
            if not self._discrete_act:
                act = np.clip(act, a_min=-1.0, a_max=1.0)
            return act
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act
