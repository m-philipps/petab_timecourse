from typing import Any, Dict, Iterable, List, Optional, TextIO, Union

import numpy as np
import pandas as pd
import petab
from petab.C import (
    CONDITION_ID,
    CONDITION_NAME,
    PARAMETER_SCALE,
    LIN,
    LOG,
    LOG10,
)

from .C import (
    NON_COMPONENT_CONDITION_LABELS,
)

class Condition(dict):
    """A PEtab condition.

    Attributes
    ----------
    id:
        The condition ID.
    components:
        The model components that are altered by the condition.
    values:
        The values that the components take.
    """

    def __init__(
            self,
            row: pd.Series,
            condition_id: str = None,
    ):
        """

        Parameters
        ----------
        row:
            A row from a PEtab conditions table.
        """
        components = {
            component: value
            for component, value in row.items()
            if component not in NON_COMPONENT_CONDITION_LABELS
        }
        components = {
            component: components[component]
            for component in sorted(components)
        }
        super().__init__(components)
        if condition_id is None:
            condition_id = row.name
        self.condition_id = condition_id
        #self.name = row.get(CONDITION_NAME, None)

    @property
    def components(self):
        return list(self.keys())

    def as_series(self):
        return pd.Series(data={
            CONDITION_ID: self.condition_id,
            CONDITION_NAME: self.name,
            **self.items(),
        })


UNSCALE = 'unscale'
SCALE = 'scale'
LOG_E_10 = np.log(10)


def gradient_chain_rule_factor_log(parameter: float):
    return parameter


def gradient_chain_rule_factor_log10(parameter: float):
    return parameter * LOG_E_10


def unscale_log_gradient(value: float, parameter: float):
    return value / gradient_chain_rule_factor_log(parameter)


def unscale_log10_gradient(value: float, parameter: float):
    return value / gradient_chain_rule_factor_log10(parameter)


def scale_log_gradient(value: float, parameter: float):
    return value * gradient_chain_rule_factor_log(parameter)


def scale_log10_gradient(value: float, parameter: float):
    return value * gradient_chain_rule_factor_log10(parameter)


def rescale_gradient_identity(value: float, parameter: float):
    return value


gradient_rescaling_methods = {
    UNSCALE: {
        LIN: rescale_gradient_identity,
        LOG: unscale_log_gradient,
        LOG10: unscale_log10_gradient,
    },
    SCALE: {
        LIN: rescale_gradient_identity,
        LOG: scale_log_gradient,
        LOG10: scale_log10_gradient,
    },
}


def rescale_state_sensitivities(
    state_sensitivities: np.ndarray,
    petab_problem: petab.problem,
    parameter_ids: List[str],
    parameters0: List[float],
    parameters1: List[float] = None,
    scales0: List[str] = None,
    flatten: bool = False,
) -> np.ndarray:
    """Rescale state sensitivities with respect to new parameters.

    Helper method to ensure correct sensitivities when passing
    gradient information between simulations of timecourse periods.

    Args:
        state_sensitivities:
            An array of state sensitivities, with dimensions
            "Number of states x Number of parameters".
        petab_problem:
            The PEtab problem that specifies the parameter scales.
        parameter_ids:
            The parameter IDs corresponding to the dimension
            "Number of parameters" in `state_sensitivities`.
        parameters0:
            The unscaled parameters that were used to compute the state
            sensitivities.
        parameters1:
            The unscaled parameters that will be used for the next simulated
            timecourse period.
        scales0:
            The parameter scales that `parameters0` were used on. Defaults to the
            scales in the PEtab problem.
        flatten:
            Whether to flatten the returned result.

    Returns:
        State sensitivity values as if they had been computed with the parameters
        used for the next simulated timecourse period.
    """
    scales1 = [
        petab_problem.parameter_df.loc[parameter_id, PARAMETER_SCALE]
        for parameter_id in parameter_ids
    ]

    # Default to both parameter vectors being used on the same scale.
    if scales0 is None:
        scales0 = scales1
    # Default to using the original parameters. Can still change state sensitivities
    # if `scales0` is specified and differs from the PEtab problem.
    if parameters1 is None:
        parameters1 = parameters0

    rescaled_state_sensitivities = np.empty(state_sensitivities.shape)
    rescaled_state_sensitivities[:] = np.nan
    for parameter_index, (
        state_sensitivity_vector,
        scale0,
        parameter0,
        scale1,
        parameter1,
    ) in enumerate(zip(
        state_sensitivities,
        scales0,
        parameters0,
        scales1,
        parameters1,
    )):
        for state_index in range(state_sensitivity_vector.size):
            unscaled_state_sensitivity = gradient_rescaling_methods[UNSCALE][scale0](
                value=state_sensitivity_vector[state_index],
                parameter=parameter0,
            )
            rescaled_state_sensitivity = gradient_rescaling_methods[SCALE][scale1](
                value=unscaled_state_sensitivity,
                parameter=parameter1,
            )
            rescaled_state_sensitivities[parameter_index, state_index] = \
                rescaled_state_sensitivity

    if np.isnan(rescaled_state_sensitivities).any():
        raise ValueError(
            'Error while rescaling state sensitivities. Some were not set and remain '
            f'NaN. Result:\n{rescaled_state_sensitivities}'
        )

    if flatten:
        rescaled_state_sensitivities = rescaled_state_sensitivities.flatten()

    return rescaled_state_sensitivities
