from itertools import chain
from typing import Dict, Sequence, Tuple

import amici
from amici.petab_import import import_petab_problem
from amici.petab_objective import (
    create_edatas,
    simulate_petab,
)
from more_itertools import one
import petab

from .misc import (
    get_timecourse,
    subset_petab_problem,
)

def simulate_timecourse(
        parent_petab_problem: petab.Problem,
        timecourse_id: str,
        problem_parameters: Dict[str, float] = None,
        parameter_mapping = None,
        sensi_orders: Tuple[int, ...] = (0,),
):
    timecourse = get_timecourse(
        petab_problem=parent_petab_problem,
        timecourse_id=timecourse_id,
    )
    petab_problems = subset_petab_problem(
        petab_problem=parent_petab_problem,
        timecourse_id=timecourse_id,
    )
    results = []
    for index, petab_problem in enumerate(petab_problems):
        amici_model = import_petab_problem(petab_problem)
        amici_model.setT0(float(timecourse[index][0]))

        amici_edatas = create_edatas(amici_model, petab_problem)
        if problem_parameters is not None:
            assert parameter_mapping is not None
            amici.parameter_mapping.fill_in_parameters(
                edatas=amici_edatas,
                problem_parameters=problem_parameters,
                scaled_parameters=True,
                parameter_mapping=parameter_mapping,
                amici_model=amici_model,
            )
        #amici_model.setParameterById(problem_parameters)
        #parameters = [
        #    problem_parameters[parameter_id]
        #    for parameter_id in amici_model.getParameterIds()
        #]
        #amici_edatas = create_edatas(amici_model, petab_problem)
        #one(amici_edatas).parameters = parameters

        if results:
            one(amici_edatas).x0  = one(results[-1]['rdatas']).x[-1].flatten()
            one(amici_edatas).sx0 = one(results[-1]['rdatas']).sx[-1].flatten()

        amici_solver = amici_model.getSolver()
        amici_solver.setSensitivityOrder(amici.SensitivityOrder_first)
        amici_solver.setSensitivityMethod(amici.SensitivityMethod_forward)

        #print(one(amici_edatas).parameters)
        #import functools
        #sp = lambda x: simulate_petab(
        #    petab_problem=petab_problem,
        #    amici_model=amici_model,
        #    solver=amici_solver,
        #    problem_parameters=x,
        #)
        ##breakpoint()
        results.append(simulate_petab(
            petab_problem=petab_problem,
            amici_model=amici_model,
            solver=amici_solver,
            edatas=amici_edatas,
            problem_parameters=problem_parameters,
        ))
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
