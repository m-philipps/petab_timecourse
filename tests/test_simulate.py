"""
# Try with custom `simulate_petab` first.
*. TODO Remove timecourse measurements from main PEtab problem
*. TODO Generate subset PEtab problems, one per timecourse
1. [x] Subset PEtab data/problem by timepoints in timecourse.
2. [x] Replace conditions table with timecourse timepoint condition
  - OR remove and use `setFixedParametersById`?
3. Remove timecourse parameters from parameters? Or expect that they are not there?
  - TODO currently a "bug" in yaml2sbml?
4. [x] Add dummy data to the PEtab problems such that timecourse timepoints are
output by the model.
5. [x] Simulate each section with `simulate_petab`
6. [x] Stitch together for full simulation
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
    X = [1 for _ in T]
    measurement_df = pd.DataFrame(data={
        'observableId': 'observable_x_',
        'simulationConditionId': TIMECOURSE_ID,
        'time': [f'{t:.1f}' for t in T],
        'measurement': X,
    })
    return petab.get_measurement_df(measurement_df)


def test_subset_petab_problem():
    petab_problem = petab.Problem.from_yaml(str(petab_yaml_path()))
    petab_problem.timecourse_df = get_timecourse_df()

    # Remove a section of measurements that includes the start/end timepoint
    # of consecutive timecourse pieces, such that dummy data should be added
    # at the start/end timepoints of the respective timecourse pieces, during
    # subsetting.
    measurement_df = get_measurement_df()
    timecourse = petab_timecourse.parse_timecourse_string(
        petab_problem.timecourse_df.loc[TIMECOURSE_ID][TIMECOURSE]
    )
    t0 = float(timecourse[0][0])
    t1 = float(timecourse[1][0])
    t2 = float(timecourse[2][0])
    mid_t0_t1 = t0 + (t1 - t0) / 2
    mid_t1_t2 = t1 + (t2 - t1) / 2
    measurement_df = measurement_df[
        ~measurement_df[TIME].astype(float).between(mid_t0_t1, mid_t1_t2)
    ]
    petab_problem.measurement_df = measurement_df

    petab_problem.condition_df = get_condition_df()
    petab_problems = petab_timecourse.subset_petab_problem(
        petab_problem,
        TIMECOURSE_ID,
    )

    for index, (t, condition_id) in enumerate(timecourse):
        if index < len(timecourse) - 1:
            for dummy_key, dummy_value in {
                    #OBSERVABLE_ID: DUMMY_OBSERVABLE_ID,
                    OBSERVABLE_FORMULA: DUMMY_MEASUREMENT,
                    NOISE_FORMULA: DUMMY_NOISE,
            }.items():
                # Dummy observable has been added.
                assert (
                    petab_problems[index]
                    .observable_df
                    .loc[DUMMY_OBSERVABLE_ID]
                    #.iloc[-1, :]  # dummy should be added as last row
                    [dummy_key]
                    == dummy_value
                )
            for dummy_key, dummy_value in {
                    OBSERVABLE_ID: DUMMY_OBSERVABLE_ID,
                    SIMULATION_CONDITION_ID: TIMECOURSE_ID,
                    MEASUREMENT: DUMMY_MEASUREMENT,
                    TIME: float(timecourse[index+1][0]),
            }.items():
                # Dummy measurements have been added.
                assert (
                    petab_problems[index]
                    .measurement_df
                    .iloc[-1, :]  # dummy should be added as last row
                    [dummy_key]
                    == dummy_value
                )
            # The simulation condition has been correctly set.
            assert (
                petab_problems[index].condition_df.loc[TIMECOURSE_ID] ==
                petab_problems[index].condition_df.loc[condition_id]
            ).all()
            # Only timecourse-specific measurements remain in the problem.
            assert (
                petab_problems[index].measurement_df[SIMULATION_CONDITION_ID]
                == TIMECOURSE_ID
            ).all()
            # No timecourse section-specific parameters exist in the PEtab
            # problem parameter table.
            assert all([
                component_id not in petab_problems[index].parameter_df.index
                for component_id in \
                    petab_problems[index].condition_df.loc[TIMECOURSE_ID].index
            ])


def test_simulate():
    parent_petab_problem = petab.Problem.from_yaml(str(petab_yaml_path()))
    parent_petab_problem.measurement_df = get_measurement_df()
    parent_petab_problem.timecourse_df = get_timecourse_df()
    parent_petab_problem.condition_df = get_condition_df()

    # The model is imported once, from the "parent" PEtab problem that contains
    # all information.
    # FIXME switched to the first of the subset PEtab problems. Presumably the
    #       first problem has a condition that sets the values of all
    #       parameters in the timecourse.
    # TODO might there be an issue with simulating some timecourses if the
    #      model isn't reimported for each timecourse section?
    #      may need to e.g. set fixed parameters for each timecourse condition
    #_ = import_petab_problem(petab_problems[0])

    results = petab_timecourse.simulate_timecourse(
        parent_petab_problem=parent_petab_problem,
        timecourse_id=TIMECOURSE_ID,
    )

    # Collect
    # FIXME magic number `0`
    x_ = list(chain.from_iterable([
        rdata.x[:, 0].flatten()
        for result in results
        for rdata in result['rdatas']
    ]))

    # Collect simulated forward sensitivities.
    sx_ = list(chain.from_iterable([
        rdata.sx[:, 0, :].flatten()
        for result in results
        for rdata in result['rdatas']
    ]))

    T = list(chain.from_iterable([
        rdata.ts
        for result in results
        for rdata in result['rdatas']
    ]))
    analytical_x_  = [np.round(get_analytical_x_(t), 5)  for t in T]
    analytical_sx_ = [np.round(get_analytical_sx_(t), 5) for t in T]

    # The state (x_) trajectory is correct.
    assert np.isclose(x_, analytical_x_).all()
    # The state (x_) forward sensitivity w.r.t. the parameter (p_) is correct.
    assert np.isclose(sx_, analytical_sx_).all()
