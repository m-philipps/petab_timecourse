from itertools import chain
from typing import Any, Dict, Sequence, Tuple, Union
import warnings

import amici
from amici.petab_import import import_petab_problem
from amici.petab_objective import (
    create_edatas,
    simulate_petab,
)
from more_itertools import one
import petab
from petab.C import CONDITION_NAME
from pypesto.objective.amici_util import create_identity_parameter_mapping

from .misc import (
    get_timecourse,
    subset_petab_problem,
)

def simulate_timecourse(
        parent_petab_problem: petab.Problem,
        timecourse_id: str,
        solver_settings: Dict[str, Any],
        problem_parameters: Dict[str, float] = None,
        parameter_mapping = None,
        sensi_orders: Tuple[int, ...] = (0,1),
        # FIXME Dict typehint
        initial_states: Union[Dict, Tuple[float, ...]] = None,
        model_settings: Dict[str, Any] = None,
        #solver_customizer: Callable[[amici.Solver], None] = None
):
    """
    Solver settings are a required attribute to ensure these are not
    forgotten...

    To use a default solver, supply an empty dictionary (`solver_settings={}`).

    initial_states:
        FIXME
        Can supply a dictionary or a tuple.
    """
    timecourse = get_timecourse(
        petab_problem=parent_petab_problem,
        timecourse_id=timecourse_id,
    )
    petab_problems = subset_petab_problem(
        petab_problem=parent_petab_problem,
        timecourse_id=timecourse_id,
    )
    #print([petab_problem.measurement_df for petab_problem in petab_problems])
    results = []
    #amici_model = import_petab_problem(petab_problems[0])
    for index, petab_problem in enumerate(petab_problems):
        if petab_problem.measurement_df.empty:
            # FIXME resolve this properly. Is it due to timecourse pieces
            #       that are defined to occur after the last measured time
            #       point?
            break
        # FIXME uncomment, remove global `amici_model = ...` above


        # Replace a controlled parameter with the ID of its timepoint-specific
        # control parameter (hack to ensure sensitivities are computed correctly)
        #print(1)
        replace_timecourse_parameters = {
            k: v
            for k, v in petab_problem.condition_df.loc[timecourse_id].items()
            if isinstance(v, str) and k != CONDITION_NAME
        }
        petab_problem.condition_df.drop(
            replace_timecourse_parameters,
            axis=1,
            inplace=True,
        )
        for old_parameter_id, new_parameter_id in replace_timecourse_parameters.items():
            if old_parameter_id in problem_parameters:
                raise ValueError(
                    'A timecourse parameter was assigned a value via '
                    '`problem_parameters`, but should receive its value '
                    'from the timecourse information.'
                )
            if new_parameter_id not in problem_parameters:
                raise ValueError(
                    'Please supply a value for estimated timecourse '
                    'replacement parameters (parameter IDs in the rows) '
                    'of the condition table of a PEtab Timecourse problem.'
                )
            problem_parameters[old_parameter_id] = \
                problem_parameters[new_parameter_id]
            # Duplicate control parameter in parameter_df, new index value is the
            # controlled parameter, such that sensitivities are output for the
            # controlled parameter (and can be interpreted as the sensitivities for
            # the control parameter)
            petab_problem.parameter_df.loc[old_parameter_id] = \
                petab_problem.parameter_df.loc[new_parameter_id]
        #print(2)

        amici_model = import_petab_problem(petab_problem)
        amici_model.setT0(float(timecourse[index][0]))
        #if initial_states is not None:
        #    breakpoint()
        #print(f'\n\nTime: {amici_model.t0()}\n\n')

        amici_edatas = create_edatas(amici_model, petab_problem)
        if problem_parameters is not None:
            # FIXME temp fix to add into parameters that AMICI automatically
            #       sets to be estimated from the SBML model (that weren't
            #       fixed like parameters in a condition table)
            model_parameters = dict(zip(
                amici_model.getParameterIds(),
                amici_model.getParameters()
            ))

            #assert parameter_mapping is not None
            #print(problem_parameters)
            #print(petab_problem.measurement_df)
            if parameter_mapping is None:
                parameter_mapping = \
                    create_identity_parameter_mapping(amici_model, 1)
            # Remove parameters from problem parameters if they are already
            # specified by the timecourse.
            # May break if a problem parameter is named `'conditionName'`.
            subset_problem_parameters = {
                parameter_id: parameter_value
                for parameter_id, parameter_value in {
                    **model_parameters,
                    **problem_parameters
                }.items()
                #for parameter_id, parameter_value in problem_parameters.items()
                if parameter_id not in petab_problem.condition_df.columns
            }
            removed_problem_parameters = \
                set(problem_parameters).difference(subset_problem_parameters)
            warnings.warn(
                'The following parameters were removed from the supplied '
                '`problem_parameters`, as they are already specified by the '
                f'timecourse: {removed_problem_parameters}'
            )

            #notpositive = {
            #    k: v
            #    for k, v in subset_problem_parameters.items()
            #    if v <= 0
            #}
            #print(f'not positive parameters: {notpositive}')  # FIXME

            amici.parameter_mapping.fill_in_parameters(
                edatas=amici_edatas,
                problem_parameters=subset_problem_parameters,
                scaled_parameters=True,
                parameter_mapping=parameter_mapping,
                amici_model=amici_model,
            )
        else:
            subset_problem_parameters = None
        #amici_model.setParameterById(problem_parameters)
        #parameters = [
        #    problem_parameters[parameter_id]
        #    for parameter_id in amici_model.getParameterIds()
        #]
        #amici_edatas = create_edatas(amici_model, petab_problem)
        #one(amici_edatas).parameters = parameters
        #print(3)

        if results:
            one(amici_edatas).x0  = one(results[-1]['rdatas']).x[-1].flatten()
            one(amici_edatas).sx0 = one(results[-1]['rdatas']).sx[-1].flatten()
        elif initial_states is not None:
            # TODO untested
            #print(initial_states)
            #print(amici_model.getStateIds())
            if isinstance(initial_states, dict):
                indexed_initial_states = [
                    initial_states[state_id]
                    for state_id in amici_model.getStateIds()
                ]
            else:
                indexed_initial_states = initial_states
            #print(indexed_initial_states)
            #print(amici_model.getStateIds())
            #print(4)
            one(amici_edatas).x0 = indexed_initial_states
            #print(5)

        amici_solver = amici_model.getSolver()
        # TODO allow a user to specify these settings
        if model_settings is not None:
            for setter, value in model_settings.items():
                getattr(amici_model, setter)(value)
        if solver_settings is not None:
            for setter, value in solver_settings.items():
                getattr(amici_solver, setter)(value)
        #print(4)
        #amici_solver.setSensitivityOrder(amici.SensitivityOrder_first)
        #amici_solver.setSensitivityMethod(amici.SensitivityMethod_forward)
        #amici_solver.setMaxSteps(int(1e6))

        #solver_settings = {
        #    'setSensitivityOrder': amici.SensitivityOrder_first,
        #    'setSensitivityMethod': amici.SensitivityMethod_forward,
        #    'setMaxSteps': int(1e6),
        #    'setMaxTime': 60,
        #}

        #amici_solver.setAbsoluteTolerance(1e-8)
        #amici_solver.setRelativeTolerance(1e-6)
        #amici_solver.setAbsoluteToleranceFSA(1e-8)
        #amici_solver.setRelativeToleranceFSA(1e-6)

        #print(one(amici_edatas).parameters)
        #import functools
        #sp = lambda x: simulate_petab(
        #    petab_problem=petab_problem,
        #    amici_model=amici_model,
        #    solver=amici_solver,
        #    problem_parameters=x,
        #)
        results.append(simulate_petab(
            petab_problem=petab_problem,
            amici_model=amici_model,
            solver=amici_solver,
            edatas=amici_edatas,
            problem_parameters=subset_problem_parameters,
        ))
        #print(5)

        #print(problem_parameters)
        #print(results)
        if initial_states is not None:
            pass
        for old_parameter_id, new_parameter_id in replace_timecourse_parameters.items():
            results[-1]['sllh'][new_parameter_id] = \
                results[-1]['sllh'][old_parameter_id]
            # Was artifically added, so remove now
            del problem_parameters[old_parameter_id]
    return results


# Collect
def collect_x(results):
    return list(chain.from_iterable([
        rdata.x[:, 0].flatten()
        for result in results
        for rdata in result['rdatas']
    ]))

# Collect simulated forward sensitivities.
def collect_sx(results):
    return list(chain.from_iterable([
        rdata.sx[:, 0, :].flatten()
        for result in results
        for rdata in result['rdatas']
    ]))

def collect_t(results):
    return list(chain.from_iterable([
        rdata.ts
        for result in results
        for rdata in result['rdatas']
    ]))
