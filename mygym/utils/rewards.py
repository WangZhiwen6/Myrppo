"""Implementation of reward functions."""


from datetime import datetime
from math import exp
from typing import Any, Dict, List, Tuple, Union

from .constants import LOG_REWARD_LEVEL, YEAR
from mygym.utils.logger import Logger


class BaseReward(object):

    logger = Logger().getLogger(name='REWARD',
                                level=LOG_REWARD_LEVEL)

    def __init__(self):
        """
        Base reward class.

        All reward functions should inherit from this class.

        Args:
            env (Env): Gym environment.
        """

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Method for calculating the reward function."""
        raise NotImplementedError(
            "Reward class must have a `__call__` method.")


class LinearReward(BaseReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        Linear reward function.

        It considers the energy consumption and the absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super(LinearReward, self).__init__()

        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Summer period
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Check variables to calculate reward are available
        try:
            assert all(temp_name in list(obs_dict.keys())
                       for temp_name in self.temp_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the temperature variables specified are not present in observation.')
            raise err
        try:
            assert all(energy_name in list(obs_dict.keys())
                       for energy_name in self.energy_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the energy variables specified are not present in observation.')
            raise err

        # Energy calculation
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)

        # Comfort violation calculation
        total_temp_violation, temp_violations = self._get_temperature_violation(
            obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward(
            energy_penalty, comfort_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'reward_weight': self.W_energy,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> Tuple[float,
                                                                 List[float]]:
        """Calculate the total energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            Tuple[float, List[float]]: Total energy consumed (sum of variables) and List with energy consumed in each energy variable.
        """

        energy_values = [
            v for k, v in obs_dict.items() if k in self.energy_names]

        # The total energy is the sum of energies
        total_energy = sum(energy_values)

        return total_energy, energy_values

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
            Tuple[float, List[float]]: Total temperature violation (ºC) and list with temperature violation in each zone.
        """

        month = obs_dict['month']
        day = obs_dict['day_of_month']
        year = YEAR
        current_dt = datetime(int(year), int(month), int(day))

        # Periods
        summer_start_date = datetime(
            int(year),
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            int(year),
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        temp_values = [
            v for k, v in obs_dict.items() if k in self.temp_names]
        total_temp_violation = 0.0
        temp_violations = []
        for T in temp_values:
            if T < temp_range[0] or T > temp_range[1]:
                temp_violation = min(
                    abs(temp_range[0] - T), abs(T - temp_range[1]))
                temp_violations.append(temp_violation)
                total_temp_violation += temp_violation

        return total_temp_violation, temp_violations

    def _get_energy_penalty(self, energy_values: List[float]) -> float:
        """Calculate the negative absolute energy penalty based on energy values

        Args:
            energy_values (List[float]): Energy values

        Returns:
            float: Negative absolute energy penalty value
        """
        energy_penalty = -sum(energy_values)
        return energy_penalty

    def _get_comfort_penalty(self, temp_violations: List[float]) -> float:
        """Calculate the negative absolute comfort penalty based on temperature violation values

        Args:
            temp_violations (List[float]): Temperature violation values

        Returns:
            float: Negative absolute comfort penalty value
        """
        comfort_penalty = -sum(temp_violations)
        return comfort_penalty

    def _get_reward(self, energy_penalty: float,
                    comfort_penalty: float) -> Tuple[float, float, float]:
        """It calculates reward value using the negative absolute comfort and energy penalty calculates previously.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float,float,float]: total reward calculated, reward term for energy, reward term for comfort.
        """
        energy_term = self.lambda_energy * self.W_energy * energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term


class ExpReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super(ExpReward, self).__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

    def _get_comfort_penalty(self, temp_violations: List[float]) -> float:
        """Calculate the negative absolute comfort penalty based on temperature violation values, using an exponential concept when temperature violation > 0.

        Args:
            temp_violations (List[float]): Temperature violation values

        Returns:
            float: Negative absolute comfort penalty value
        """
        comfort_penalty = -sum(list(map(lambda temp_violation: exp(
            temp_violation) if temp_violation > 0 else 0, temp_violations)))
        return comfort_penalty


class HourlyLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        default_energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        range_comfort_hours: tuple = (9, 19),
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            default_energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            range_comfort_hours (tuple, optional): Hours where thermal comfort is considered. Defaults to (9, 19).
        """

        super(HourlyLinearReward, self).__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            default_energy_weight,
            lambda_energy,
            lambda_temperature
        )

        # Reward parameters
        self.range_comfort_hours = range_comfort_hours
        self.default_energy_weight = default_energy_weight

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Check variables to calculate reward are available
        try:
            assert all(temp_name in list(obs_dict.keys())
                       for temp_name in self.temp_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the temperature variables specified are not present in observation.')
            raise err
        try:
            assert all(energy_name in list(obs_dict.keys())
                       for energy_name in self.energy_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the energy variables specified are not present in observation.')
            raise err

        # Energy calculation
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)

        # Comfort violation calculation
        total_temp_violation, temp_violations = self._get_temperature_violation(
            obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)

        # Determine reward weight depending on the hour
        hour = obs_dict['hour']
        if hour >= self.range_comfort_hours[0] and hour <= self.range_comfort_hours[1]:
            self.W_energy = self.default_energy_weight
        else:
            self.W_energy = 1.0

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward(
            energy_penalty, comfort_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'reward_weight': self.W_energy,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation
        }

        return reward, reward_terms


class NormalizedLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        max_energy_penalty: float = 8,
        max_comfort_penalty: float = 12,
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            default_energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            range_comfort_hours (tuple, optional): Hours where thermal comfort is considered. Defaults to (9, 19).
        """

        super(NormalizedLinearReward, self).__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight
        )

        # Reward parameters
        self.max_energy_penalty = max_energy_penalty
        self.max_comfort_penalty = max_comfort_penalty

    def _get_reward(self, energy_penalty: float,
                    comfort_penalty: float) -> Tuple[float, float, float]:
        """It calculates reward value using energy consumption and grades of temperature out of comfort range. Aplying normalization

        Args:
            energy (float): Negative absolute energy penalty value.
            comfort (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float,float,float]: total reward calculated, reward term for energy and reward term for comfort.
        """
        # Update max energy and comfort
        self.max_energy_penalty = max(self.max_energy_penalty, energy_penalty)
        self.max_comfort_penalty = max(
            self.max_comfort_penalty, comfort_penalty)
        # Calculate normalization
        energy_norm = 0 if energy_penalty == 0 else energy_penalty / self.max_energy_penalty
        comfort_norm = 0 if comfort_penalty == 0 else comfort_penalty / self.max_comfort_penalty
        # Calculate reward terms with norm values
        energy_term = self.W_energy * energy_norm
        comfort_term = (1 - self.W_energy) * comfort_norm
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term
    



class MultiZoneReward(BaseReward):

    def __init__(
        self,
        energy_variables: List[str],
        temperature_and_setpoints_conf: Dict[str, str],
        comfort_threshold: float = 0.5,
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        A linear reward function for environments with different comfort ranges in each zone. Instead of having
        a fixed and common comfort range for the entire building, each zone has its own comfort range, which is
        directly obtained from the setpoints established in the building. This function is designed for buildings
        where temperature setpoints are not controlled directly but rather used as targets to be achieved, while
        other actuators are controlled to reach these setpoints. A setpoint observation variable can be assigned
        per zone if it is available in the specific building. It is also possible to assign the same setpoint
        variable to multiple air temperature zones.

        Args:
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            temperature_and_setpoints_conf (Dict[str, str]): Dictionary with the temperature variable name (key) and the setpoint variable name (value) of the observation space.
            comfort_threshold (float, optional): Comfort threshold for temperature range (+/-). Defaults to 0.5.
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Name of the variables
        self.energy_names = energy_variables
        self.comfort_configuration = temperature_and_setpoints_conf
        self.comfort_threshold = comfort_threshold

        # Reward parameters
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature
        self.comfort_ranges = {}

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
            'comfort_threshold': self.comfort_threshold
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
           List[float]: List with temperature violation (ºC) in each zone.
        """
        # Calculate current comfort range for each zone
        self._get_comfort_ranges(obs_dict)

        temp_violations = [
            max(0, min(abs(T - comfort_range[0]), abs(T - comfort_range[1])))
            if T < comfort_range[0] or T > comfort_range[1] else 0
            for temp_var, comfort_range in self.comfort_ranges.items()
            if (T := obs_dict[temp_var])
        ]

        return temp_violations

    def _get_comfort_ranges(
            self, obs_dict: Dict[str, Any]):
        """Calculate the comfort range for each zone in the current observation.

        Returns:
            Dict[str, Tuple[float, float]]: Comfort range for each zone.
        """
        # Calculate current comfort range for each zone
        self.comfort_ranges = {
            temp_var: (setpoint - self.comfort_threshold, setpoint + self.comfort_threshold)
            for temp_var, setpoint_var in self.comfort_configuration.items()
            if (setpoint := obs_dict[setpoint_var]) is not None
        }

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term


import numpy as np

class myyreward(ExpReward):
    """
    Enhanced ExpReward class that directly handles action vectors containing setpoints.
    Specifically designed for 56 thermal zones with action space containing heating and cooling setpoints.
    Action vector format: [heating_1, cooling_1, heating_2, cooling_2, ..., heating_56, cooling_56]
    """

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.4,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        setpoint_weight: float = 0.2,
        lambda_setpoint: float = 1.0,
        setpoint_change_threshold: float = 0.5,
        num_zones: int = 56,
        max_setpoint_change_penalty: float = 10.0
    ):
        """
        Enhanced ExpReward with action-based setpoint change penalty.

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start.
            summer_final (Tuple[int,int]): Summer session tuple with month and day end.
            energy_weight (float): Weight given to the energy term. Defaults to 0.4.
            lambda_energy (float): Constant for removing dimensions from power. Defaults to 1.0.
            lambda_temperature (float): Constant for removing dimensions from temperature. Defaults to 1.0.
            setpoint_weight (float): Weight given to the setpoint penalty term. Defaults to 0.2.
            lambda_setpoint (float): Constant for setpoint penalty scaling. Defaults to 1.0.
            setpoint_change_threshold (float): Minimum change to be considered significant. Defaults to 0.5.
            num_zones (int): Number of thermal zones. Defaults to 56.
            max_setpoint_change_penalty (float): Maximum penalty for setpoint changes. Defaults to 10.0.
        """

        super(myyreward, self).__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

        # Setpoint penalty parameters
        self.setpoint_weight = setpoint_weight
        self.lambda_setpoint = lambda_setpoint
        self.setpoint_change_threshold = setpoint_change_threshold
        self.num_zones = num_zones
        self.max_setpoint_change_penalty = max_setpoint_change_penalty
        
        # Store previous action vector (setpoints)
        # Initialize with default values: heating=20, cooling=24 for all zones
        self.previous_actions = None
        
        # Store setpoint change statistics for monitoring
        self.setpoint_changes_count = 0
        self.total_penalty_accumulated = 0.0
        
        self.logger.info(f'ExpReward with action-based setpoint penalty initialized for {num_zones} zones.')

    def set_current_action(self, action: Union[List[float], np.ndarray]):
        """
        Set the current action vector (setpoints) for penalty calculation.
        Should be called before calling the reward function.
        
        Args:
            action: Action vector with format [heating_1, cooling_1, heating_2, cooling_2, ...]
        """
        self.current_actions = np.array(action) if not isinstance(action, np.ndarray) else action

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function including action-based setpoint change penalty.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Get standard reward components from parent class
        base_reward, base_terms = super(myyreward, self).__call__(obs_dict)
        
        # Calculate setpoint change penalty if we have current and previous actions
        if hasattr(self, 'current_actions') and self.previous_actions is not None:
            setpoint_penalty, setpoint_changes = self._get_action_setpoint_change_penalty()
        else:
            setpoint_penalty = 0.0
            setpoint_changes = 0
        
        # Update previous actions for next step
        if hasattr(self, 'current_actions'):
            self.previous_actions = self.current_actions.copy()
        
        # Calculate setpoint penalty term
        setpoint_term = self.lambda_setpoint * self.setpoint_weight * setpoint_penalty
        
        # Calculate final reward with rebalanced weights
        # Adjust base reward weight to account for new setpoint term
        adjusted_base_weight = 1.0 - self.setpoint_weight
        final_reward = adjusted_base_weight * base_reward + setpoint_term
        
        # Update reward terms
        enhanced_terms = base_terms.copy()
        enhanced_terms.update({
            'setpoint_term': setpoint_term,
            'setpoint_penalty': setpoint_penalty,
            'setpoint_changes': setpoint_changes,
            'setpoint_weight': self.setpoint_weight,
            'total_setpoint_changes': self.setpoint_changes_count,
            'total_penalty_accumulated': self.total_penalty_accumulated,
            'adjusted_base_weight': adjusted_base_weight
        })

        return final_reward, enhanced_terms

    def _get_action_setpoint_change_penalty(self) -> Tuple[float, int]:
        """
        Calculate penalty based on setpoint changes in action vector.
        Action format: [heating_1, cooling_1, heating_2, cooling_2, ..., heating_56, cooling_56]
        
        Returns:
            Tuple[float, int]: Penalty value and number of significant changes
        """
        total_penalty = 0.0
        change_count = 0
        
        # Calculate changes for all setpoints
        for i in range(len(self.current_actions)):
            setpoint_change = abs(self.current_actions[i] - self.previous_actions[i])
            
            if setpoint_change > self.setpoint_change_threshold:
                # Apply exponential penalty that increases with change magnitude
                # Cap the penalty to avoid extreme values
                raw_penalty = exp(setpoint_change)
                capped_penalty = min(raw_penalty, self.max_setpoint_change_penalty)
                penalty = -capped_penalty
                
                total_penalty += penalty
                change_count += 1
        
        # Apply additional penalty for simultaneous changes in many zones
        if change_count > self.num_zones * 0.3:  # More than 30% of zones changed
            simultaneous_change_penalty = -exp(change_count / self.num_zones)
            total_penalty += simultaneous_change_penalty
        
        # Update statistics
        self.setpoint_changes_count += change_count
        self.total_penalty_accumulated += abs(total_penalty)
        
        return total_penalty, change_count

    def reset_setpoint_history(self):
        """Reset the setpoint history. Useful at the beginning of new episodes."""
        self.previous_actions = None
        self.setpoint_changes_count = 0
        self.total_penalty_accumulated = 0.0







import math
class Myreward(LinearReward):
    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        co2_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = [6, 1],
        summer_final: Tuple[int, int] = [9, 30],
        energy_weight: float = 0.32,
        temp_violation_weight: float = 0.38,
        co2_weight: float = 0.3,
        lambda_energy: float = 1e-5,
        lambda_temperature: float = 1,
        lambda_co2: float = 1,
        range_comfort_co2: Tuple[int, int] = [0, 800],
        co2_threshold: float = 800
    ):
        super(Myreward).__init__()
        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables
        self.co2_names = co2_variables
        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.energy_weight = energy_weight
        self.temp_weight =  temp_violation_weight
        self.co2_weight = co2_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature
        self.lambda_co2 = lambda_co2
        self.range_comfort_co2 = range_comfort_co2
        self.co2_threshold = co2_threshold

        # Summer period
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)


        self.logger.info('Reward function initialized.')


    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        try:
            assert all(temp_name in list(obs_dict.keys())
                       for temp_name in self.temp_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the temperature variables specified are not present in observation.')
            raise err
        try:
            assert all(energy_name in list(obs_dict.keys())
                       for energy_name in self.energy_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the energy variables specified are not present in observation.')
            raise err
        try:
            assert all(co2_name in list(obs_dict.keys())
                       for co2_name in self.co2_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the co2 variables specified are not present in observation.')
            raise err    

        # 基础计算
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)
        
        # 温度违规
        total_temp_violation, temp_violations = self._get_temperature_violation(obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)
        
        # CO2违规
        total_co2_violation, co2_violations = self._get_co2_violation(obs_dict)
        co2_penalty = self._get_co2_penalty(co2_violations)

        # Weighted sum of both terms
        reward, energy_term, comfort_term, co2_term = self._get_reward(
            energy_penalty, comfort_penalty, co2_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'co2_term': co2_term,            
            'reward_weight': self.energy_weight,
            'temp_weight': self.temp_weight,
            'co2_weight': self.co2_weight,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'abs_co2_penalty': co2_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation,
            'total_co2_violation': total_co2_violation
        }

        return reward, reward_terms
    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> Tuple[float,
                                                                 List[float]]:
        """Calculate the total energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            Tuple[float, List[float]]: Total energy consumed (sum of variables) and List with energy consumed in each energy variable.
        """

        energy_values = [
            v for k, v in obs_dict.items() if k in self.energy_names]

        # The total energy is the sum of energies
        total_energy = sum(energy_values)

        return total_energy, energy_values
    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
            Tuple[float, List[float]]: Total temperature violation (ºC) and list with temperature violation in each zone.
        """

        month = obs_dict['month']
        day = obs_dict['day_of_month']
        year = YEAR
        current_dt = datetime(int(year), int(month), int(day))

        # Periods
        summer_start_date = datetime(
            int(year),
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            int(year),
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        temp_values = [
            v for k, v in obs_dict.items() if k in self.temp_names]
        total_temp_violation = 0.0
        temp_violations = []
        sigma_squared = 2.2
        for T in temp_values:
            error = 0
            if T < temp_range[0]:
                error = temp_range[0] - T
            elif T > temp_range[1]:
                error = T - temp_range[1]
            if error > 0:
                R_temp = 10 * (1 / (math.sqrt(2) * math.sqrt(2 * math.pi))) * math.exp(-(error**2) / (2 * sigma_squared)) - 1.8
            else:
                R_temp = 0
            temp_violations.append(R_temp)
            total_temp_violation += R_temp
        # total_temp_violation = 0.0
        # temp_violations = []
        # for T in temp_values:
        #     if T < temp_range[0] or T > temp_range[1]:
        #         temp_violation = min(
        #             abs(temp_range[0] - T), abs(T - temp_range[1]))
        #         temp_violations.append(temp_violation)
        #         total_temp_violation += temp_violation

        return total_temp_violation, temp_violations
    def _get_energy_penalty(self, energy_values: List[float]) -> float:
        """Calculate the negative absolute energy penalty based on energy values

        Args:
            energy_values (List[float]): Energy values

        Returns:
            float: Negative absolute energy penalty value
        """
        energy_penalty = -sum(energy_values)
        return energy_penalty
    def _get_comfort_penalty(self, temp_violations: List[float]) -> float:
        """Calculate the negative absolute comfort penalty based on temperature violation values

        Args:
            temp_violations (List[float]): Temperature violation values

        Returns:
            float: Negative absolute comfort penalty value
        """
        comfort_penalty = sum(temp_violations)
        return comfort_penalty

    def _get_co2_violation(self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        co2_values = [
            v for k, v in obs_dict.items() if k in self.co2_names]
        co2_violations = []
        total_violation = 0.0

        # 双曲正切函数
        # for co2 in co2_values:
        #     if co2 <= 1000:
        #         R_co2 = -math.tanh(0.01 * (co2-self.co2_threshold))
        #     else:
        #         R_co2 = -10
        #     co2_violations.append(R_co2)
        #     total_violation += R_co2
        
        # swish函数
        mu = 900
        # swish函数的灵敏度参数
        beta = 0.02 
        
        for co2 in co2_values:
            if co2 > mu:
                deviation = co2 - mu
                swish_penalty = -deviation * (1 / (1 + math.exp(-beta * deviation)))
            else:
                swish_penalty = 0
            co2_violations.append(swish_penalty)
            total_violation += swish_penalty             
        return total_violation, co2_violations

    def _get_co2_penalty(self, co2_violations: List[float]) -> float:
        """CO2惩罚项"""
        co2_penalty = sum(co2_violations)
        return co2_penalty
    def _get_reward(self, energy_penalty: float,
                    comfort_penalty: float,co2_penalty: float) -> Tuple[float, float, float, float]:
        """
        It calculates reward value using the negative absolute comfort and energy penalty calculates previously.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float,float,float]: total reward calculated, reward term for energy, reward term for comfort.
        """
        energy_term = self.lambda_energy * self.energy_weight * energy_penalty
        comfort_term = self.lambda_temp * self.temp_weight * comfort_penalty
        co2_term = self.lambda_co2 * self.co2_weight * co2_penalty
        reward = energy_term + comfort_term + co2_term
        return reward, energy_term, comfort_term, co2_term