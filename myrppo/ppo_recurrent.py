from copy import deepcopy
from collections.abc import Sequence
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from myrppo.common.buffers import RolloutBuffer
from myrppo.common.callbacks import BaseCallback
from myrppo.common.on_policy_algorithm import OnPolicyAlgorithm
from myrppo.common.policies import BasePolicy
from myrppo.common.type_aliases import GymEnv, MaybeCallback, Schedule
from myrppo.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from myrppo.common.vec_env import VecEnv
from myrppo.safety_critic import CostCritic

from myrppo.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from myrppo.recurrent.policies import RecurrentActorCriticPolicy
from myrppo.recurrent.type_aliases import RNNStates
from myrppo.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy

SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")


class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_recurrent_policies`
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        use_safety_critic: bool = False,
        cost_critic_learning_rate: float = 1e-3,
        nu_learning_rate: float = 1e-3,
        cost_critic_hidden_sizes: Sequence[int] = (128, 128),
        cost_temperature_indices: Optional[Sequence[int]] = None,
        cost_temperature_min: Optional[float] = None,
        cost_temperature_max: Optional[float] = None,
        cost_unnormalize_observations: bool = True,
        cost_limit: float = 0.0,
        cost_violation_power: float = 2.0,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states = None
        self.use_safety_critic = use_safety_critic
        if cost_limit < 0.0:
            raise ValueError("cost_limit must be non-negative.")
        if cost_violation_power <= 0.0:
            raise ValueError("cost_violation_power must be positive.")
        self.cost_critic_learning_rate = cost_critic_learning_rate
        self.nu_learning_rate = nu_learning_rate
        self.cost_critic_hidden_sizes = tuple(cost_critic_hidden_sizes)
        self.cost_temperature_indices = (
            list(cost_temperature_indices)
            if cost_temperature_indices is not None
            else None
        )
        self.cost_temperature_min = cost_temperature_min
        self.cost_temperature_max = cost_temperature_max
        self.cost_unnormalize_observations = cost_unnormalize_observations
        self.cost_limit = float(cost_limit)
        self.cost_violation_power = float(cost_violation_power)
        self.cost_critic: Optional[CostCritic] = None
        self.cost_critic_optimizer: Optional[th.optim.Optimizer] = None
        self.nu: Optional[th.nn.Parameter] = None
        self.nu_optimizer: Optional[th.optim.Optimizer] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = RecurrentDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RecurrentRolloutBuffer

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )
        # 完整rollout buffer中存储LSTM状态所需的shape
        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        if self.use_safety_critic:
            self._setup_safety_critic()

    def _setup_safety_critic(self) -> None:
        if not isinstance(self.observation_space, spaces.Box):
            raise ValueError("Safety Critic currently requires a Box observation space.")

        input_dim = int(np.prod(self.observation_space.shape))
        self.cost_temperature_indices = self._resolve_cost_temperature_indices(input_dim)
        self.cost_temperature_min, self.cost_temperature_max = self._resolve_cost_temperature_bounds()

        self.cost_critic = CostCritic(input_dim=input_dim, hidden_sizes=self.cost_critic_hidden_sizes).to(self.device)
        self.cost_critic_optimizer = th.optim.Adam(self.cost_critic.parameters(), lr=self.cost_critic_learning_rate)
        self.nu = th.nn.Parameter(th.tensor(1.0, dtype=th.float32, device=self.device))
        self.nu_optimizer = th.optim.Adam([self.nu], lr=self.nu_learning_rate)

    def _get_first_env_attr(self, attr_name: str) -> Any:
        if self.env is None:
            return None
        try:
            values = self.env.get_attr(attr_name)
        except Exception:
            return None
        return values[0] if values else None

    def _resolve_cost_temperature_indices(self, input_dim: int) -> list[int]:
        if self.cost_temperature_indices is not None:
            indices = [int(index) for index in self.cost_temperature_indices]
        else:
            observation_variables = self._get_first_env_attr("observation_variables")
            reward_fn = self._get_first_env_attr("reward_fn")
            temp_names = getattr(reward_fn, "temp_names", None)

            if observation_variables is not None and temp_names:
                indices = [observation_variables.index(name) for name in temp_names]
            elif observation_variables is not None:
                indices = [
                    idx
                    for idx, name in enumerate(observation_variables)
                    if "air_temperature" in str(name).lower()
                ]
            else:
                indices = []

        if not indices:
            raise ValueError(
                "Safety Critic needs temperature indices. Pass cost_temperature_indices "
                "or use an environment exposing observation_variables/reward_fn.temp_names."
            )
        if min(indices) < 0 or max(indices) >= input_dim:
            raise ValueError(f"Invalid cost temperature indices for observation dim {input_dim}: {indices}")
        return indices

    def _resolve_cost_temperature_bounds(self) -> tuple[float, float]:
        if self.cost_temperature_min is not None and self.cost_temperature_max is not None:
            return float(self.cost_temperature_min), float(self.cost_temperature_max)

        reward_fn = self._get_first_env_attr("reward_fn")
        comfort_range = getattr(reward_fn, "range_comfort_summer", None)
        if comfort_range is None:
            comfort_range = getattr(reward_fn, "range_comfort_winter", None)
        if comfort_range is None:
            raise ValueError(
                "Safety Critic needs comfort bounds. Pass cost_temperature_min "
                "and cost_temperature_max explicitly."
            )
        return float(comfort_range[0]), float(comfort_range[1])

    def _flatten_cost_states(self, states: th.Tensor) -> th.Tensor:
        if states.ndim > 2:
            return th.flatten(states, start_dim=1)
        return states

    def _cost_label_states(self, states: th.Tensor) -> th.Tensor:
        states = self._flatten_cost_states(states)
        if not self.cost_unnormalize_observations:
            return states

        obs_rms = self._get_first_env_attr("obs_rms")
        if obs_rms is None:
            return states

        mean = th.as_tensor(obs_rms.mean, device=states.device, dtype=states.dtype).view(1, -1)
        var = th.as_tensor(obs_rms.var, device=states.device, dtype=states.dtype).view(1, -1)
        if mean.shape[1] != states.shape[1] or var.shape[1] != states.shape[1]:
            return states

        epsilon = self._get_first_env_attr("epsilon")
        epsilon = float(epsilon) if epsilon is not None else 1e-8
        return states * th.sqrt(var + epsilon) + mean

    def _temperature_violation_cost(self, states: th.Tensor) -> th.Tensor:
        if self.cost_temperature_indices is None:
            raise RuntimeError("Cost temperature indices were not initialized.")
        label_states = self._cost_label_states(states)
        indices = th.as_tensor(self.cost_temperature_indices, device=states.device, dtype=th.long)
        temperatures = label_states.index_select(dim=1, index=indices)
        lower = th.as_tensor(float(self.cost_temperature_min), device=states.device, dtype=states.dtype)
        upper = th.as_tensor(float(self.cost_temperature_max), device=states.device, dtype=states.dtype)
        violation = th.relu(lower - temperatures) + th.relu(temperatures - upper)
        if self.cost_violation_power != 1.0:
            violation = violation.pow(self.cost_violation_power)
        return violation.sum(dim=1)

    def _predict_cost_values(self, states: th.Tensor) -> th.Tensor:
        if self.cost_critic is None:
            return th.zeros(states.shape[0], device=self.device, dtype=th.float32)
        critic_param = next(self.cost_critic.parameters())
        states = self._flatten_cost_states(states).to(device=critic_param.device, dtype=critic_param.dtype)
        return self.cost_critic(states).flatten()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)
                if self.use_safety_critic:
                    cost_values = self._predict_cost_values(obs_tensor)
                else:
                    cost_values = th.zeros(env.num_envs, device=self.device, dtype=th.float32)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            next_obs = deepcopy(new_obs)
            for idx, done_ in enumerate(dones):
                terminal_obs = infos[idx].get("terminal_observation")
                if done_ and terminal_obs is not None:
                    if isinstance(next_obs, dict):
                        for key in next_obs.keys():
                            next_obs[key][idx] = terminal_obs[key]
                    else:
                        next_obs[idx] = terminal_obs

            if self.use_safety_critic:
                with th.no_grad():
                    costs = self._temperature_violation_cost(obs_as_tensor(next_obs, self.device)).cpu().numpy()
            else:
                costs = np.zeros(env.num_envs, dtype=np.float32)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value
                    if self.use_safety_critic:
                        with th.no_grad():
                            terminal_cost_value = self._predict_cost_values(terminal_obs)[0]
                        costs[idx] += self.gamma * terminal_cost_value.cpu().numpy()

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                next_obs=next_obs,
                cost=costs,
                cost_value=cost_values,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            last_obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(last_obs_tensor, lstm_states.vf, episode_starts)
            if self.use_safety_critic:
                cost_values = self._predict_cost_values(last_obs_tensor)
            else:
                cost_values = th.zeros(env.num_envs, device=self.device, dtype=th.float32)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.compute_cost_returns_and_advantage(last_cost_values=cost_values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        constrained_pg_losses = []
        clip_fractions = []
        cost_policy_losses = []
        cost_value_losses = []
        mean_costs = []
        cost_constraint_violations = []
        nu_values = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                # 价值函数裁剪
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                constrained_policy_loss = policy_loss
                if self.use_safety_critic:
                    if self.cost_critic is None or self.cost_critic_optimizer is None or self.nu_optimizer is None or self.nu is None:
                        raise RuntimeError("Safety Critic optimizers were not initialized.")

                    valid_ratio = ratio[mask]
                    valid_cost_advantages = rollout_data.cost_advantages[mask]
                    cost_policy_loss = th.mean(valid_ratio * valid_cost_advantages)

                    cost_values = self._predict_cost_values(rollout_data.observations)
                    cost_value_loss = th.mean(((rollout_data.cost_returns - cost_values) ** 2)[mask])

                    self.cost_critic_optimizer.zero_grad()
                    cost_value_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                    self.cost_critic_optimizer.step()

                    mean_cost = th.mean(rollout_data.costs[mask])
                    cost_constraint_violation = mean_cost - self.cost_limit
                    self.nu_optimizer.zero_grad()
                    loss_nu = -self.nu * cost_constraint_violation.detach()
                    loss_nu.backward()
                    self.nu_optimizer.step()
                    with th.no_grad():
                        self.nu.clamp_(min=0.0)

                    constrained_policy_loss = policy_loss + self.nu.detach() * cost_policy_loss

                    cost_policy_losses.append(cost_policy_loss.item())
                    cost_value_losses.append(cost_value_loss.item())
                    mean_costs.append(mean_cost.item())
                    cost_constraint_violations.append(cost_constraint_violation.item())
                    nu_values.append(self.nu.item())

                constrained_pg_losses.append(constrained_policy_loss.item())
                loss = constrained_policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/constrained_policy_gradient_loss", np.mean(constrained_pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if self.use_safety_critic:
            self.logger.record("train/cost_policy_loss", np.mean(cost_policy_losses))
            self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
            self.logger.record("train/thermal_violation_cost", np.mean(mean_costs))
            self.logger.record("train/cost_constraint_violation", np.mean(cost_constraint_violations))
            self.logger.record("train/safety_nu", np.mean(nu_values))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_last_lstm_states"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, torch_vars = super()._get_torch_save_params()
        if self.use_safety_critic:
            state_dicts = state_dicts + ["cost_critic", "cost_critic_optimizer", "nu_optimizer"]
            torch_vars = torch_vars + ["nu"]
        return state_dicts, torch_vars
