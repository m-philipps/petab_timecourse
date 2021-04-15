"""
1. [x] Create a custom pyPESTO objective to handle the PEtab timecourse
       simulation.
2. ...


"""
from copy import deepcopy
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence, Tuple
from warnings import warn

import amici
from amici.petab_import import import_petab_problem
from amici.petab_objective import simulate_petab
import numpy as np
import pandas as pd
import petab
from petab.C import (
    MEASUREMENT,
    NOISE_FORMULA,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    SIMULATION_CONDITION_ID,
    TIME,
)
import pytest
import yaml2sbml

import petab_timecourse
from petab_timecourse.C import (
    DUMMY_MEASUREMENT,
    DUMMY_NOISE,
    DUMMY_OBSERVABLE_ID,
    TIMECOURSE,
)

from fixture_fixed_timepoint_parameter_timecourse import (
    OUTPUT_DENSITY,
    petab_yaml_path,
    timecourse_q_,

    get_analytical_x_,
    get_analytical_sx_
)


TIMECOURSE_ID = 'timecourse1'


def get_condition_df():
    condition_df = pd.DataFrame(data={
        'conditionId': [TIMECOURSE_ID, 'q_pos', 'q_zero', 'q_neg'],
        'q_': [None, 1, 0, -1],
    })
    return petab.get_condition_df(condition_df)


def get_timecourse_df():
    timecourse_df = pd.DataFrame(data={
        'timecourseId': [TIMECOURSE_ID],
        'timecourse': '0:q_pos;10:q_zero;20:q_neg;30:q_zero;40:q_pos'
    })
    # TODO use read function to "cleanup" df/ set index?
    return petab_timecourse.get_timecourse_df(timecourse_df)


def get_measurement_df():
    T = np.linspace(0, 100, 1001)
    #X = [1 for _ in T]
    X = [get_analytical_x_(t) for t in T]
    measurement_df = pd.DataFrame(data={
        'observableId': 'observable_x_',
        'simulationConditionId': TIMECOURSE_ID,
        'time': [f'{t:.1f}' for t in T],
        'measurement': X,
    })
    return petab.get_measurement_df(measurement_df)


def test_optimize():
    parent_petab_problem = petab.Problem.from_yaml(str(petab_yaml_path()))
    parent_petab_problem.measurement_df = get_measurement_df()
    parent_petab_problem.timecourse_df = get_timecourse_df()
    parent_petab_problem.condition_df = get_condition_df()

    importer = petab_timecourse.pypesto.TimecoursePetabImporter(
        petab_problem=parent_petab_problem,
        timecourse_id=TIMECOURSE_ID,
    )
    objective = importer.create_objective()
    objective([0.1])
    breakpoint()
    # The model is imported once, from the "parent" PEtab problem that contains
    # all information.
    # FIXME switched to the first of the subset PEtab problems. Presumably the
    #       first problem has a condition that sets the values of all
    #       parameters in the timecourse.
    # TODO might there be an issue with simulating some timecourses if the
    #      model isn't reimported for each timecourse section?
    #      may need to e.g. set fixed parameters for each timecourse condition
    #_ = import_petab_problem(petab_problems[0])

    #results = petab_timecourse.simulate_timecourse(
    #    parent_petab_problem=parent_petab_problem,
    #    timecourse_id=TIMECOURSE_ID,
    #)

    ## Collect
    ## FIXME magic number `0`
    #x_ = list(chain.from_iterable([
    #    rdata.x[:, 0].flatten()
    #    for result in results
    #    for rdata in result['rdatas']
    #]))

    ## Collect simulated forward sensitivities.
    #sx_ = list(chain.from_iterable([
    #    rdata.sx[:, 0, :].flatten()
    #    for result in results
    #    for rdata in result['rdatas']
    #]))

    #T = list(chain.from_iterable([
    #    rdata.ts
    #    for result in results
    #    for rdata in result['rdatas']
    #]))
    #analytical_x_  = [np.round(get_analytical_x_(t), 5)  for t in T]
    #analytical_sx_ = [np.round(get_analytical_sx_(t), 5) for t in T]

    ## The state (x_) trajectory is correct.
    #assert np.isclose(x_, analytical_x_).all()
    ## The state (x_) forward sensitivity w.r.t. the parameter (p_) is correct.
    #assert np.isclose(sx_, analytical_sx_).all()
