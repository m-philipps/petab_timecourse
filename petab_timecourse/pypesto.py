from typing import Dict, Tuple

from amici.petab_import import import_petab_problem
import numpy as np
import petab
from pypesto.objective import ObjectiveBase
from pypesto.objective.constants import (
    MODE_FUN,
    FVAL,
    GRAD,
)
from pypesto.objective.amici_util import create_identity_parameter_mapping

from .amici import simulate_timecourse
from .misc import subset_petab_problem


class TimecoursePetabImporter():
    def __init__(self, petab_problem, timecourse_id):
        self.petab_problem = petab_problem
        self.timecourse_id = timecourse_id
        self.petab_problems = subset_petab_problem(
            petab_problem=petab_problem,
            timecourse_id=timecourse_id,
        )

    def create_objective(self):
        # FIXME assumes first model will have all required features for later
        #       models (e.g. same condition parameters etc)
        amici_model = import_petab_problem(self.petab_problems[0])
        return TimecourseObjective(
            petab_problem=self.petab_problem,
            timecourse_id=self.timecourse_id,
            x_names=list(amici_model.getParameterIds()),
            # 1 is the number of condition. Currently only 1 "condition" (1
            # timecourse) is supported.
            parameter_mapping=create_identity_parameter_mapping(amici_model, 1)
        )


class TimecourseObjective(ObjectiveBase):
    def __init__(
            self,
            petab_problem,
            timecourse_id,
            x_names,
            parameter_mapping,
        ):
        self.petab_problem = petab_problem
        self.timecourse_id = timecourse_id
        self.parameter_mapping = parameter_mapping
        super().__init__(x_names=x_names)

    def check_sensi_orders(self, sensi_orders, mode) -> bool:
        if max(sensi_orders) > 1:
            return False
        return self.check_mode(mode)

    def check_mode(self, mode) -> bool:
        if mode != MODE_FUN:
            return False
        return True

    def call_unprocessed(
            self, 
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str,
            **kwargs,
    ):
        # TODO sensi_orders, mode
        return simulate_timecourse_objective(
            self.petab_problem,
            self.timecourse_id,
            dict(zip(self.x_names, x)),
            self.parameter_mapping,
            **kwargs,
        )


def simulate_timecourse_objective(
        parent_petab_problem: petab.Problem,
        timecourse_id: str,
        problem_parameters: Dict[str, float],
        parameter_mapping,
        **kwargs,
):
    results = simulate_timecourse(
        parent_petab_problem,
        timecourse_id,
        problem_parameters=problem_parameters,
        parameter_mapping=parameter_mapping,
    )
    return_result = {
        FVAL: sum(-result['llh']  for result in results),
        # FIXME Assumes all timecourse sections have the same parameter SLLH
        GRAD: {
            k: sum(-result['sllh'][k] for result in results)
            for k in results[0]['sllh']
        }
    }
    print(results)
    print(return_result)
    #breakpoint()
    print('breaking')
    return {
        FVAL: sum(-result['llh']  for result in results),
        # FIXME Assumes all timecourse sections have the same parameter SLLH
        GRAD: {
            k: sum(-result['sllh'][k] for result in results)
            for k in results[0]['sllh']
        }
    }
