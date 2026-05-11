import os
import sys
import importlib
from datetime import datetime

# Allow `python rppo.py` direct execution while keeping relative imports.
# if __package__ in (None, ""):
#     _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#     _PARENT_DIR = os.path.dirname(_THIS_DIR)
#     _COMMON_DIR = os.path.join(_THIS_DIR, "common")
#     for _path in (_PARENT_DIR, _THIS_DIR, _COMMON_DIR):
#         if _path not in sys.path:
#             sys.path.insert(0, _path)
#     __package__ = "myrppo"

_PACKAGE_NAME = __package__

# Allow `python rppo.py` direct execution while keeping relative imports.
if __package__ in (None, ""):
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PARENT_DIR = os.path.dirname(_THIS_DIR)
    _COMMON_DIR = os.path.join(_THIS_DIR, "common")
    for _path in (_PARENT_DIR, _THIS_DIR, _COMMON_DIR):
        if _path not in sys.path:
            sys.path.insert(0, _path)
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
from myrppo.common.logger import HumanOutputFormat
from myrppo.common.logger import Logger as SB3Logger
from myrppo.ppo_recurrent import RecurrentPPO
from myrppo.structured_actions import TopologyActionProjectionWrapper
from myrppo.thermal_zone_graph import THERMAL_ZONE_EDGES

num_zones = 56
COMFORT_TEMPERATURE_MIN = 20.0
COMFORT_TEMPERATURE_MAX = 24.0
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


def build_temperature_variables() -> list[str]:
    return [
        f"{zone.lower().replace(' ', '_')}_air_temperature"
        for zone in thermal_zones
    ]


def main() -> None:
    new_action_space = gym.spaces.Box(
        low=np.array([18, 22] * 56, dtype=np.float32),
        high=np.array([22, 26] * 56, dtype=np.float32),
        dtype=np.float32,
    )

    new_meters = {
        "EnergyHeating": "DistrictHeating:Facility",
        "EnergyCooling": "DistrictCooling:Facility",
    }
    new_reward_kwargs = {
        "temperature_variables": build_temperature_variables(),
        "energy_variables": list(new_meters.keys()),
        "range_comfort_winter": (COMFORT_TEMPERATURE_MIN, COMFORT_TEMPERATURE_MAX),
        "range_comfort_summer": (COMFORT_TEMPERATURE_MIN, COMFORT_TEMPERATURE_MAX),
        "energy_weight": 1.0,
        "lambda_energy": 1e-6,
        "lambda_temperature": 0.0,
    }

    environment = "Eplus-1-mixed-continuous-stochastic-v1"
    episodes = 20
    experiment_date = datetime.today().strftime("%Y-%m-%d-%H_%M")
    experiment_name = f"RPPO-{environment}-episodes-{episodes}_{experiment_date}"

    extra_params = {"timesteps_per_hour": 6, "runperiod": (1, 7, 2006, 31, 7, 2006)}

    env = gym.make(
        environment,
        env_name=experiment_name,
        config_params=extra_params,
        actuators=build_actuators(),
        meters=new_meters,
        action_space=new_action_space,
        reward=LinearReward,
        reward_kwargs=new_reward_kwargs,
    )

    env = NormalizeObservation(env)
    env = NormalizeAction(env)


    # 拓扑约束
    TOPOLOGY_SMOOTHING_LAMBDA = 0.20
    env = TopologyActionProjectionWrapper(
        env,
        zone_edges=THERMAL_ZONE_EDGES,
        num_zones=num_zones,
        smoothing_lambda=TOPOLOGY_SMOOTHING_LAMBDA,
    )



    env = NormalizeReward(env)
    env = LoggerWrapper(env)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        batch_size=256,
        n_steps=2048,
        verbose=1,
        use_barrier=True,
        barrier_lambda=0.01,
        barrier_learning_rate=3e-4,
        nu_learning_rate=3e-4,
        barrier_temperature_min=COMFORT_TEMPERATURE_MIN,
        barrier_temperature_max=COMFORT_TEMPERATURE_MAX,
        policy_kwargs=dict(lstm_hidden_size=256, 
                           n_lstm_layers=3,
                           shared_lstm=False,
                           enable_critic_lstm=True,),
    )

    callbacks = []
    # logger = SB3Logger(
    #     folder=None, output_formats=[HumanOutputFormat(sys.stdout, max_length=120)]
    # )
    # model.set_logger(logger)

    timesteps = episodes * (env.get_wrapper_attr("timestep_per_episode") - 1)
    # callbacks.append(LoggerCallback())
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
