import copy
from typing import Any, Dict, Tuple

from amici.petab_import import import_petab_problem
from more_itertools import one
import numpy as np
import petab
from pypesto.objective import ObjectiveBase
from pypesto.C import (
    MODE_FUN,
    FVAL,
    GRAD,
)
from pypesto.objective.amici_util import create_identity_parameter_mapping
from pypesto.petab import PetabImporter

from .amici import simulate_timecourse
from .misc import subset_petab_problem
from .C import (
    FINAL_STATES,
    RESULTS,
)


class PetabTimecourseImporter(PetabImporter):
    def __init__(self, petab_problem, timecourse_id):
        self.petab_problem = petab_problem
        self.timecourse_id = timecourse_id
        self.petab_problems = subset_petab_problem(
            petab_problem=petab_problem,
            timecourse_id=timecourse_id,
        )
        # TODO add additional things from PetabImporter.__init__

    def create_objective(self, **kwargs):
        # FIXME assumes first model will have all required features for later
        #       models (e.g. same condition parameters etc)
        # FIXME move to `__init__`, use `self.petab_problem`?
        amici_model = import_petab_problem(self.petab_problems[0])
        return PetabTimecourseObjective(
            petab_problem=self.petab_problem,
            timecourse_id=self.timecourse_id,
            #x_names=list(amici_model.getParameterIds()),
            x_names=list(self.petab_problem.parameter_df.index),
            # 1 is the number of condition. Currently only 1 "condition" (1
            # timecourse) is supported.
            parameter_mapping=create_identity_parameter_mapping(amici_model,1),
            **kwargs,
        )

    #def create_problem(
    #    self,
    #    objective: TimecourseObjective = None,
    #    x_guesses: Optional[Iterable[float]] = None,
    #    **kwargs,
    #) -> Problem:
    #    """Create a :class:`pypesto.Problem`.
    #    Parameters
    #    ----------
    #    objective:
    #        Objective as created by `create_objective`.
    #    x_guesses:
    #        Guesses for the parameter values, shape (g, dim), where g denotes
    #        the number of guesses. These are used as start points in the
    #        optimization.
    #    **kwargs:
    #        Additional key word arguments passed on to the objective,
    #        if not provided.
    #    Returns
    #    -------
    #    problem:
    #        A :class:`pypesto.Problem` for the objective.
    #    """
    #    if objective is None:
    #        objective = self.create_objective(**kwargs)

    #    prior = self.create_prior()

    #    if prior is not None:
    #        objective = AggregatedObjective([objective, prior])

    #    x_scales = \
    #        [self.petab_problem.parameter_df.loc[x_id, petab.PARAMETER_SCALE]
    #            for x_id in self.petab_problem.x_ids]

    #    problem = Problem(
    #        objective=objective,
    #        lb=self.petab_problem.lb_scaled,
    #        ub=self.petab_problem.ub_scaled,
    #        x_fixed_indices=self.petab_problem.x_fixed_indices,
    #        x_fixed_vals=self.petab_problem.x_nominal_fixed_scaled,
    #        x_guesses=x_guesses,
    #        startpoint_method=self.create_startpoint_method(),
    #        x_names=self.petab_problem.x_ids,
    #        x_scales=x_scales,
    #        x_priors_defs=prior)

    #    return problem




class PetabTimecourseObjective(ObjectiveBase):
    def __init__(
            self,
            petab_problem,
            timecourse_id,
            x_names,
            parameter_mapping,
            solver_settings: Dict[str, Any] = None,
            model_settings: Dict[str, Any] = None,
            default_parameters: Dict[str, float] = None,
            initial_states = None,  # TODO typehint, condition-specific
        ):
        self.petab_problem = petab_problem
        self.timecourse_id = timecourse_id
        self.parameter_mapping = parameter_mapping
        self.solver_settings = solver_settings
        self.model_settings = model_settings
        self.default_parameters = default_parameters
        self.initial_states = initial_states
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
            return_all: bool = False,
            **kwargs,
    ):
        # TODO mode
        return simulate_timecourse_objective(
            parent_petab_problem=self.petab_problem,
            timecourse_id=self.timecourse_id,
            problem_parameters=dict(zip(self.x_names, x)),
            parameter_mapping=self.parameter_mapping,
            sensi_orders=sensi_orders,
            return_all=return_all,
            solver_settings=self.solver_settings,
            model_settings=self.model_settings,
            initial_states=self.initial_states,
            **kwargs,
        )

    def __deepcopy__(self, memodict: Dict = None) -> 'PetabTimecourseObjective':
        other = self.__class__.__new__(self.__class__)

        for key in self.__dict__:
            other.__dict__[key] = copy.deepcopy(self.__dict__[key])

        return other


# FIXME Deprecate
TimecoursePetabImporter = PetabTimecourseImporter
TimecourseObjective = PetabTimecourseObjective


def simulate_timecourse_objective(
        parent_petab_problem: petab.Problem,
        timecourse_id: str,
        problem_parameters: Dict[str, float],
        parameter_mapping,
        sensi_orders: Tuple[int, ...] = (0,),
        return_all: bool = False,
        max_abs_grad: float = None,
        **kwargs,
):
    #unscaled_problem_parameters = unscale_parameters(
    #    scaled_parameters=problem_parameters,
    #    petab_problem=parent_petab_problem,
    #)
    unscaled_problem_parameters = parent_petab_problem.unscale_parameters(
        scaled_parameters=problem_parameters,
    )

    #print('in')
    results = simulate_timecourse(
        parent_petab_problem,
        timecourse_id,
        problem_parameters=unscaled_problem_parameters,
        parameter_mapping=parameter_mapping,
        sensi_orders=sensi_orders,
        **kwargs,
    )
    #print('out')
    if kwargs.get('initial_states', None) is not None:
        pass
        #breakpoint()

    #sensitivity_parameter_ids = results[0]['sllh'].keys()
    sensitivity_parameter_ids = problem_parameters.keys()
    #print('l1')
    #print(problem_parameters)
    #breakpoint()
    #sensitivity_parameter_ids = parent_petab_problem.x_ids
    for result in results:
        if result['sllh'].keys() != sensitivity_parameter_ids:
            # FIXME reimplement so this still holds?
            #raise NotImplementedError(
            #    'All conditions must provide sensitivities for the same set '
            #    'of parameters.'
            #)
            pass
    #print('l2')
    accumulated_result = {
        FVAL: sum(-result['llh']  for result in results),
        GRAD: [
            sum(
                -result['sllh'][k]
                for result in results
                if k in result['sllh']
            )
            for k in sensitivity_parameter_ids
            #if k in result['sllh']
        ]
        #GRAD: {
        #    k: sum(-result['sllh'][k] for result in results)
        #    for k in sensitivity_parameter_ids
        #}
    }
    #print('l3')
    if return_all:
        # TODO magic constant
        accumulated_result[FINAL_STATES] = \
            one(results[-1]['rdatas']).x[-1].flatten()
        #accumulated_result[RESULTS] = results
    #print('l4')
    #print(accumulated_result)
    #print(accumulated_result)
    #breakpoint()
    #print(accumulated_result)

    if max_abs_grad is not None:
        pass
        #accumulated_result[GRAD] = list(np.nan_to_num(
        #    np.array(accumulated_result[GRAD]),
        #    nan=np.nan,
        #    posinf=max_abs_grad,
        #    neginf=-max_abs_grad,
        #))
    #print(accumulated_result)

    return accumulated_result
