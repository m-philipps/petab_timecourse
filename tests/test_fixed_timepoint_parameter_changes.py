"""Test model with parameter changes to constant values at fixed timepoints.

Tests for expected state and state (forward) sensitivity trajectories.

d/dt x_ = p_ * q_
p_ = 1
q_ is a timecourse
for t in [ 0, 10), q_ =  1
    t in [10, 20), q_ =  0
    t in [20, 30), q_ = -1
    t in [30, 40), q_ =  0
    t in [40, 50], q_ =  1

Analytical solutions
--------------------
The following should be evaluated according to the timecourse for q_.
 x_ = p_ * q_ * t
sx_ =      q_ * t  # sensitivity of x_ w.r.t. p_
"""
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
import pytest
import yaml2sbml


TEST_ID = 'fixed_timepoint_parameter_changes'
p_ = 1
d_q_ = np.array((
#   ( t, q_),  # `t` is time, `q_` is the value that the parameter takes.
    ( 0,  1),
    (10,  0),
    (20, -1),
    (30,  0),
    (40,  1),
    (50,  None),  # None indicates the end of the simulation.
))

# FIXME switch to PEtab measurements
OUTPUT_DENSITY = 10  # Number of output timepoints per time unit.


def get_analytical_x_(
        t: float,
        d_q_: Sequence[Tuple[float, float]] = d_q_,
        p_ = p_,
) -> float:
    """The expected value for $x_$.

    Parameters
    ----------
    t:
        The current time.
    d_q_:
        The values that `q_` changes to as time progresses, as a sequence of
        tuples. In each tuple, the first value is a timepoint, and the second
        value is the value that `q_` changes to at that timepoint. A `q_` value
        of `None` indicates the final timepoint.
    """
    x_ = 0
    for row_index, (t_start, q_) in enumerate(d_q_):
        if q_ is None:
            warn(
                'Expected maximum simulation time possibly exceeded requested '
                f'simulation time. Expected: {t_start}. Requested: {t}.'
            )
            break
        t_end = d_q_[row_index + 1, 0]
        if t <= t_end:
            x_ += p_ * q_ * (t - t_start)
            break
        x_ += p_ * q_ * (t_end - t_start)
    return x_


def get_analytical_sx_(
        t: float,
        d_q_: Sequence[Tuple[float, float]] = d_q_
) -> float:
    """The expected sensitivity for x_ w.r.t. p_.

    Parameters
    ----------
    t:
        The current time.
    d_q_:
        The values that `q_` changes to as time progresses, as a sequence of
        tuples. In each tuple, the first value is a timepoint, and the second
        value is the value that `q_` changes to at that timepoint. A `q_` value
        of `None` indicates the final timepoint.
    """
    sx_ = 0
    for row_index, (t_start, q_) in enumerate(d_q_):
        if q_ is None:
            warn(
                'Expected maximum simulation time possibly exceeded requested '
                f'simulation time. Expected: {t_start}. Requested: {t}.'
            )
            break
        t_end = d_q_[row_index+1, 0]
        if t <= t_end:
            sx_ += q_ * (t - t_start)
            break
        sx_ += q_ * (t_end - t_start)

    return sx_


@pytest.fixture
def yaml2sbml_model_string():
    """Defines the model used by the test."""
    return """
odes:
    - stateId: x_
      rightHandSide: p_ * q_
      initialValue: 0

parameters:
    - parameterId: p_
      nominalValue: 1
      parameterScale: lin
      lowerBound: 0.1
      upperBound: 10
      estimate: 1

    - parameterId: q_
      nominalValue: 1
      parameterScale: lin
      lowerBound: 0.1
      upperBound: 10
      estimate: 0

observables:
    - observableId: observable_x_
      observableFormula: x_
      observableTransformation: lin
      noiseFormula: 1
      noiseDistribution: normal

    - observableId: observable_q_
      observableFormula: q_
      observableTransformation: lin
      noiseFormula: 1
      noiseDistribution: normal

conditions:
    - conditionId: condition1
"""


@pytest.fixture
def petab_yaml_path(yaml2sbml_model_string):
    petab_path = Path('output') / TEST_ID
    petab_path.mkdir(parents=True, exist_ok=True)

    petab_yaml_filename = 'petab.yaml'
    measurement_filename = 'measurements.tsv'

    with NamedTemporaryFile('w') as yaml2sbml_file:
        yaml2sbml_file.write(yaml2sbml_model_string)
        yaml2sbml_file.flush()
        
        yaml2sbml.yaml2petab(
            yaml_dir=yaml2sbml_file.name,
            output_dir=str(petab_path),
            sbml_name=TEST_ID,
            petab_yaml_name=petab_yaml_filename,
            measurement_table_name=measurement_filename,
        )
    
    # Dummy measurements
    # FIXME replace with expected values...
    # FIXME use timecourse ID as simulation condition ID
    T = np.linspace(0, 100, 1001)
    X = [1 for _ in T]
    measurement_df = pd.DataFrame(data={
        'observableId': 'observable_x_',
        'simulationConditionId': 'condition1',
        'time': [f'{t:.1f}' for t in T],
        'measurement': X,
    })
    measurement_df = petab.get_measurement_df(measurement_df)
    petab.write_measurement_df(
        measurement_df,
        str(petab_path / measurement_filename),
    )

    return petab_path / petab_yaml_filename


def test_model(petab_yaml_path):
    petab_problem = petab.Problem.from_yaml(str(petab_yaml_path))
    model = import_petab_problem(petab_problem, force_compile=False)  # FIXME force_compile True
    solver = model.getSolver()
    solver.setSensitivityOrder(1)
    solver.setSensitivityMethod(1)
    
    rdatas = []
    rdata = None
    for row_index, (t_start, q_) in enumerate(d_q_):
        if q_ is None:
            break
        t_end = d_q_[row_index+1, 0]
        model.setTimepoints(
            np.linspace(t_start, t_end, OUTPUT_DENSITY*(t_end-t_start) + 1)
        )
        model.setParameterById({'q_': q_})
    
        # Continue the next sub-simulation from the previous simulation.
        if rdatas:
            model.setT0(t_start)
            model.setInitialStates(rdatas[-1].x[-1])
            # 0th column is sensitivity of x_ w.r.t. p_
            # 1st column is sensitivity of x_ w.r.t. q_
            model.setInitialStateSensitivities(rdatas[-1].sx[-1][:, 0])
        rdatas.append(amici.runAmiciSimulation(model, solver))
    
    # Collect
    x_ = list(chain.from_iterable([
        rdata.sx[:, 0, :].flatten()
        for rdata in rdatas
    ]))

    # Collect simulated forward sensitivities.
    sx_ = list(chain.from_iterable([
        rdata.sx[:, 0, :].flatten()
        for rdata in rdatas
    ]))

    T = list(chain.from_iterable([rdata.ts for rdata in rdatas]))
    analytical_x_  = [np.round(get_analytical_x_(t), 5)  for t in T]
    analytical_sx_ = [np.round(get_analytical_sx_(t), 5) for t in T]

    # The state (x_) trajectory is correct.
    assert np.isclose(x_, analytical_x_).all()
    # The state (x_) forward sensitivity w.r.t. the parameter (p_) is correct.
    assert np.isclose(sx_, analytical_sx_).all()
