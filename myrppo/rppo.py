import os
import sys
import importlib
import argparse
import json
from datetime import datetime
from typing import Any, Callable

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
_COMMON_DIR = os.path.join(_THIS_DIR, "common")

# Keep local top-level imports such as `common.logger` resolvable both when this
# file is run as a script and when it is imported as `myrppo.rppo`.
for _path in (_PARENT_DIR, _THIS_DIR, _COMMON_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Allow `python rppo.py` direct execution while keeping relative imports.
if __package__ in (None, ""):
    if _COMMON_DIR not in sys.path:
        sys.path.insert(0, _COMMON_DIR)
    _PACKAGE_NAME = os.path.basename(_THIS_DIR)
    __package__ = _PACKAGE_NAME
else:
    _PACKAGE_NAME = __package__

# Unify callback module identity for absolute/relative imports inside local SB3 copy.
# Some files import `callbacks` while others import `<package>.common.callbacks`.
if _PACKAGE_NAME and "callbacks" not in sys.modules:
    sys.modules["callbacks"] = importlib.import_module(f"{_PACKAGE_NAME}.common.callbacks")


import gymnasium as gym
import numpy as np
import mygym
import mygym.utils.gcloud as gcloud
from gymnasium.wrappers.stateful_reward import NormalizeReward
from mygym.utils.callbacks import *
from mygym.utils.constants import *
from mygym.utils.logger import CSVLogger, WandBOutputFormat
from mygym.utils.rewards import *
from mygym.utils.wrappers import *

from myrppo.common.callbacks import CallbackList
from myrppo.common.evaluation import evaluate_policy as evaluate_vec_policy
from myrppo.common.logger import HumanOutputFormat
from myrppo.common.logger import Logger as SB3Logger
from myrppo.ppo_recurrent import RecurrentPPO

num_zones = 56
thermal_zones = (
    "THERMAL ZONE: HALL-1-1",
    "THERMAL ZONE: HALL-1-10",
    "THERMAL ZONE: HALL-1-11",
    "THERMAL ZONE: HALL-1-12",
    "THERMAL ZONE: HALL-1-13",
    "THERMAL ZONE: HALL-1-2",
    "THERMAL ZONE: HALL-1-3",
    "THERMAL ZONE: HALL-1-4",
    "THERMAL ZONE: HALL-1-5",
    "THERMAL ZONE: HALL-1-6",
    "THERMAL ZONE: HALL-1-7",
    "THERMAL ZONE: HALL-1-8",
    "THERMAL ZONE: HALL-1-9",
    "THERMAL ZONE: HALL-2-1",
    "THERMAL ZONE: HALL-2-2",
    "THERMAL ZONE: HALL-2-3",
    "THERMAL ZONE: HALL-3-1",
    "THERMAL ZONE: HALL-3-2",
    "THERMAL ZONE: HALL-3-3",
    "THERMAL ZONE: HALL-4-1",
    "THERMAL ZONE: HALL-4-2",
    "THERMAL ZONE: HALL-4-3",
    "THERMAL ZONE: HALL-4-4",
    "THERMAL ZONE: P1-1-COMMERCE 1",
    "THERMAL ZONE: P1-1-DINING 1",
    "THERMAL ZONE: P1-1-OFFICE 1",
    "THERMAL ZONE: P1-1-OFFICE 2",
    "THERMAL ZONE: P1-10-RESTROOM 1",
    "THERMAL ZONE: P1-11-COMMERCE 1",
    "THERMAL ZONE: P1-11-COMMERCE 2",
    "THERMAL ZONE: P1-11-OFFICE 1",
    "THERMAL ZONE: P1-11-OFFICE 2",
    "THERMAL ZONE: P1-2-BREAKROOM 1",
    "THERMAL ZONE: P1-2-DINING 1",
    "THERMAL ZONE: P1-2-RESTROOM 1",
    "THERMAL ZONE: P1-3-COMMERCE 1",
    "THERMAL ZONE: P1-4-COMMERCE 1",
    "THERMAL ZONE: P1-4-DINING 1",
    "THERMAL ZONE: P1-5-COMMERCE 1",
    "THERMAL ZONE: P1-6-COMMERCE 1",
    "THERMAL ZONE: P1-7-DINING 1",
    "THERMAL ZONE: P1-7-OFFICE 1",
    "THERMAL ZONE: P1-8-RESTROOM 1",
    "THERMAL ZONE: P1-9-COMMERCE 1",
    "THERMAL ZONE: P2-1-COMMERCE 2",
    "THERMAL ZONE: P2-1-COMMERCE 3",
    "THERMAL ZONE: P2-2-RESTROOM 1",
    "THERMAL ZONE: P3-1-COMMERCE 1",
    "THERMAL ZONE: P3-1-DINING 1",
    "THERMAL ZONE: P3-2-DINING 1",
    "THERMAL ZONE: P3-2-RESTROOM 1",
    "THERMAL ZONE: P4-1-COMMERCE 1",
    "THERMAL ZONE: P4-2-COMMERCE 1",
    "THERMAL ZONE: P4-2-COMMERCE 2",
    "THERMAL ZONE: P4-3-COMMERCE 1",
    "THERMAL ZONE: P4-3-COMMERCE 2",
)


BATCH_SIZE_CHOICES = (64, 128, 256, 512)

RIME_PARAMETER_SPECS = (
    ("learning_rate", 1e-5, 1e-2, "log_float"),
    ("ent_coef", 0.01, 0.5, "float"),
    ("clip_range", 0.1, 0.3, "float"),
    ("gamma", 0.95, 0.99, "float"),
    ("gae_lambda", 0.9, 1.0, "float"),
    ("batch_size", BATCH_SIZE_CHOICES, None, "choice"),
)

DEFAULT_RPPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "batch_size": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RPPO and optionally tune its hyperparameters with RIME."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--no-rime", action="store_true", help="Skip RIME and use default RPPO hyperparameters.")
    parser.add_argument("--rime-population", type=int, default=4)
    parser.add_argument("--rime-iterations", type=int, default=2)
    parser.add_argument("--rime-search-episodes", type=int, default=1)
    parser.add_argument("--rime-eval-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_actuators() -> dict:
    new_actuators = {}
    for i in range(1, num_zones + 1):
        heating_key = f"{thermal_zones[i - 1]}-Heating"
        cooling_key = f"{thermal_zones[i - 1]}-Cooling"
        new_actuators[heating_key] = (
            "Zone Temperature Control",
            "Heating Setpoint",
            thermal_zones[i - 1],
        )
        new_actuators[cooling_key] = (
            "Zone Temperature Control",
            "Cooling Setpoint",
            thermal_zones[i - 1],
        )
    return new_actuators


def build_action_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.array([18, 22] * num_zones, dtype=np.float32),
        high=np.array([22, 26] * num_zones, dtype=np.float32),
        dtype=np.float32,
    )


def build_meters() -> dict[str, str]:
    return {
        "EnergyHeating": "DistrictHeating:Facility",
        "EnergyCooling": "DistrictCooling:Facility",
    }


def make_training_env(environment: str, experiment_name: str, extra_params: dict[str, Any]) -> gym.Env:
    env = gym.make(
        environment,
        env_name=experiment_name,
        config_params=extra_params,
        actuators=build_actuators(),
        meters=build_meters(),
        action_space=build_action_space(),
    )

    env = NormalizeObservation(env)
    env = NormalizeAction(env)
    env = NormalizeReward(env)
    env = LoggerWrapper(env)
    return env


def make_model(
    env: gym.Env,
    hyperparams: dict[str, Any],
    *,
    verbose: int,
    seed: int | None,
) -> RecurrentPPO:
    return RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=1024,
        verbose=verbose,
        seed=seed,
        policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=3),
        **hyperparams,
    )


def decode_rime_position(position: np.ndarray) -> dict[str, Any]:
    hyperparams: dict[str, Any] = {}
    for index, (name, low, high, value_type) in enumerate(RIME_PARAMETER_SPECS):
        value = float(np.clip(position[index], 0.0, 1.0))
        if value_type == "choice":
            choices = tuple(low)
            choice_index = min(int(value * len(choices)), len(choices) - 1)
            hyperparams[name] = choices[choice_index]
        elif value_type == "log_float":
            log_low = np.log10(low)
            log_high = np.log10(high)
            hyperparams[name] = float(10 ** (log_low + value * (log_high - log_low)))
        elif value_type == "int":
            hyperparams[name] = int(round(low + value * (high - low)))
        else:
            hyperparams[name] = float(low + value * (high - low))
    return hyperparams


def format_hyperparams(hyperparams: dict[str, Any]) -> str:
    ordered_values = []
    for name, _, _, _ in RIME_PARAMETER_SPECS:
        value = hyperparams[name]
        if isinstance(value, float):
            ordered_values.append(f"{name}={value:.6g}")
        else:
            ordered_values.append(f"{name}={value}")
    return ", ".join(ordered_values)


def save_best_hyperparams(
    path: str,
    *,
    environment: str,
    experiment_name: str,
    hyperparams: dict[str, Any],
    best_score: float | None,
    args: argparse.Namespace,
) -> None:
    payload = {
        "algorithm": "RIME-RPPO" if not args.no_rime else "RPPO",
        "environment": environment,
        "experiment_name": experiment_name,
        "best_score": best_score,
        "best_hyperparams": hyperparams,
        "rime": {
            "enabled": not args.no_rime,
            "population": args.rime_population,
            "iterations": args.rime_iterations,
            "search_episodes": args.rime_search_episodes,
            "eval_episodes": args.rime_eval_episodes,
            "seed": args.seed,
        },
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def normalize_fitness(fitness: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(fitness)
    if not finite_mask.any():
        return np.zeros_like(fitness, dtype=np.float64)

    finite_values = fitness[finite_mask]
    min_fitness = float(np.min(finite_values))
    max_fitness = float(np.max(finite_values))
    if np.isclose(max_fitness, min_fitness):
        normalized = np.zeros_like(fitness, dtype=np.float64)
        normalized[finite_mask] = 1.0
        return normalized

    normalized = np.zeros_like(fitness, dtype=np.float64)
    normalized[finite_mask] = (fitness[finite_mask] - min_fitness) / (max_fitness - min_fitness)
    return normalized


def rime_optimize(
    fitness_fn: Callable[[dict[str, Any], int, int], float],
    *,
    population_size: int,
    iterations: int,
    seed: int | None,
) -> tuple[dict[str, Any], float]:
    population_size = max(1, population_size)
    iterations = max(0, iterations)
    dimension = len(RIME_PARAMETER_SPECS)
    rng = np.random.default_rng(seed)

    population = rng.uniform(0.0, 1.0, size=(population_size, dimension))
    fitness = np.full(population_size, -np.inf, dtype=np.float64)

    for agent_index in range(population_size):
        candidate = decode_rime_position(population[agent_index])
        fitness[agent_index] = fitness_fn(candidate, 0, agent_index)

    best_index = int(np.argmax(fitness))
    best_position = population[best_index].copy()
    best_fitness = float(fitness[best_index])

    for iteration in range(1, iterations + 1):
        fitness_weights = normalize_fitness(fitness)
        new_population = population.copy()
        new_fitness = fitness.copy()
        progress = iteration / max(iterations, 1)
        soft_rime_probability = np.sqrt(progress)
        soft_rime_scale = np.cos(np.pi * iteration / (10.0 * max(iterations, 1))) * (1.0 - progress)

        for agent_index in range(population_size):
            candidate_position = population[agent_index].copy()
            for dim in range(dimension):
                if rng.random() < soft_rime_probability:
                    rime_factor = (2.0 * rng.random() - 1.0) * soft_rime_scale * rng.random()
                    candidate_position[dim] = best_position[dim] + rime_factor
                if rng.random() < fitness_weights[agent_index]:
                    candidate_position[dim] = best_position[dim]

            candidate_position = np.clip(candidate_position, 0.0, 1.0)
            candidate = decode_rime_position(candidate_position)
            candidate_fitness = fitness_fn(candidate, iteration, agent_index)
            if candidate_fitness >= fitness[agent_index]:
                new_population[agent_index] = candidate_position
                new_fitness[agent_index] = candidate_fitness

        population = new_population
        fitness = new_fitness
        iteration_best_index = int(np.argmax(fitness))
        if fitness[iteration_best_index] >= best_fitness:
            best_position = population[iteration_best_index].copy()
            best_fitness = float(fitness[iteration_best_index])

        print(
            "[RIME] iteration "
            f"{iteration}/{iterations}, best_score={best_fitness:.6g}, "
            f"best_params: {format_hyperparams(decode_rime_position(best_position))}"
        )

    if not np.isfinite(best_fitness):
        raise RuntimeError("RIME failed to find a finite fitness value.")
    return decode_rime_position(best_position), best_fitness


def evaluate_rppo_hyperparams(
    hyperparams: dict[str, Any],
    iteration: int,
    agent_index: int,
    *,
    environment: str,
    experiment_prefix: str,
    extra_params: dict[str, Any],
    search_episodes: int,
    eval_episodes: int,
    seed: int | None,
) -> float:
    experiment_name = f"{experiment_prefix}-rime-i{iteration:02d}-a{agent_index:02d}"
    env = make_training_env(environment, experiment_name, extra_params)
    model: RecurrentPPO | None = None
    try:
        model = make_model(env, hyperparams, verbose=0, seed=seed)
        model.set_logger(SB3Logger(folder=None, output_formats=[]))
        timesteps_per_episode = env.get_wrapper_attr("timestep_per_episode") - 1
        search_timesteps = max(1, search_episodes * timesteps_per_episode)
        model.learn(total_timesteps=search_timesteps, callback=None, log_interval=100)

        vec_env = model.get_env()
        if vec_env is None:
            raise RuntimeError("RPPO model has no evaluation environment.")
        episode_rewards, _ = evaluate_vec_policy(
            model,
            vec_env,
            n_eval_episodes=max(1, eval_episodes),
            deterministic=True,
            return_episode_rewards=True,
            warn=False,
        )
        score = float(np.mean(episode_rewards))
        print(
            "[RIME] "
            f"iteration={iteration}, agent={agent_index}, score={score:.6g}, "
            f"{format_hyperparams(hyperparams)}"
        )
        return score
    except Exception as exc:
        print(
            "[RIME] "
            f"iteration={iteration}, agent={agent_index} failed: {exc}. "
            f"{format_hyperparams(hyperparams)}"
        )
        return -np.inf
    finally:
        if model is not None and model.get_env() is not None:
            model.get_env().close()
        else:
            env.close()


def main() -> None:
    args = parse_args()

    environment = "Eplus-1-mixed-continuous-stochastic-v1"
    episodes = args.episodes
    experiment_date = datetime.today().strftime("%Y-%m-%d-%H_%M")
    experiment_name = f"RIME-RPPO-{environment}-episodes-{episodes}_{experiment_date}"

    extra_params = {"timesteps_per_hour": 6, "runperiod": (1, 7, 2006, 31, 7, 2006)}

    hyperparams = DEFAULT_RPPO_HYPERPARAMS.copy()
    best_score = None
    if not args.no_rime:
        experiment_prefix = f"RIME-search-{environment}-{experiment_date}"
        hyperparams, best_score = rime_optimize(
            lambda candidate, iteration, agent_index: evaluate_rppo_hyperparams(
                candidate,
                iteration,
                agent_index,
                environment=environment,
                experiment_prefix=experiment_prefix,
                extra_params=extra_params,
                search_episodes=max(1, args.rime_search_episodes),
                eval_episodes=max(1, args.rime_eval_episodes),
                seed=args.seed,
            ),
            population_size=args.rime_population,
            iterations=args.rime_iterations,
            seed=args.seed,
        )
        print(f"[RIME] selected score={best_score:.6g}, params: {format_hyperparams(hyperparams)}")
    else:
        experiment_name = f"RPPO-{environment}-episodes-{episodes}_{experiment_date}"
        print(f"[RIME] skipped, using params: {format_hyperparams(hyperparams)}")

    env = make_training_env(environment, experiment_name, extra_params)

    model = make_model(env, hyperparams, verbose=1, seed=args.seed)

    callbacks = []
    logger = SB3Logger(
        folder=None, output_formats=[HumanOutputFormat(sys.stdout, max_length=120)]
    )
    model.set_logger(logger)

    timesteps = episodes * (env.get_wrapper_attr("timestep_per_episode") - 1)
    output_dir = str(env.get_wrapper_attr("timestep_per_episode"))
    os.makedirs(output_dir, exist_ok=True)
    best_hyperparams_path = os.path.join(output_dir, "best_hyperparams.json")
    save_best_hyperparams(
        best_hyperparams_path,
        environment=environment,
        experiment_name=experiment_name,
        hyperparams=hyperparams,
        best_score=best_score,
        args=args,
    )
    print(f"[RIME] best hyperparameters saved to {best_hyperparams_path}")

    callbacks.append(LoggerCallback())
    callback = CallbackList(callbacks)

    model.learn(total_timesteps=timesteps, callback=callback, log_interval=100)

    model.save(f"{env.get_wrapper_attr('timestep_per_episode')}/{experiment_name}")

    if hasattr(env, "mean") and hasattr(env, "var"):
        training_mean = env.get_wrapper_attr("mean")
        training_var = env.get_wrapper_attr("var")
        _ = (training_mean, training_var)

    env.close()


if __name__ == "__main__":
    main()
