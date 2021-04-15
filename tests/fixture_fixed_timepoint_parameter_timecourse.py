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

#from petab_timecourse import 


TEST_ID = 'fixed_timepoint_parameter_timecourse'
p_ = 1
timecourse_q_ = np.array((
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
        timecourse_q_: Sequence[Tuple[float, float]] = timecourse_q_,
        p_: float = p_,
) -> float:
    """The expected value for $x_$.

    Parameters
    ----------
    t:
        The current time.
    timecourse_q_:
        The values that `q_` changes to as time progresses, as a sequence of
        tuples. In each tuple, the first value is a timepoint, and the second
        value is the value that `q_` changes to at that timepoint. A `q_` value
        of `None` indicates the final timepoint.
    """
    x_ = 0
    for row_index, (t_start, q_) in enumerate(timecourse_q_):
        if q_ is None:
            # FIXME support None in middle of timecourses, as continuation of
            # previous... currently None is expected to occur at the end
            # If past the last timepoint, continue with the last parameter
            # value.
            previous_q_ = timecourse_q_[row_index - 1][1]
            x_ += p_ * previous_q_ * (t - t_start)
            break
        t_end = timecourse_q_[row_index + 1, 0]
        if t <= t_end:
            x_ += p_ * q_ * (t - t_start)
            break
        x_ += p_ * q_ * (t_end - t_start)
    return x_


def get_analytical_sx_(
        t: float,
        timecourse_q_: Sequence[Tuple[float, float]] = timecourse_q_
) -> float:
    """The expected sensitivity for x_ w.r.t. p_.

    Parameters
    ----------
    t:
        The current time.
    timecourse_q_:
        The values that `q_` changes to as time progresses, as a sequence of
        tuples. In each tuple, the first value is a timepoint, and the second
        value is the value that `q_` changes to at that timepoint. A `q_` value
        of `None` indicates the final timepoint.
    """
    sx_ = 0
    for row_index, (t_start, q_) in enumerate(timecourse_q_):
        if q_ is None:
            # FIXME support None in middle of timecourses, as continuation of
            # previous... currently None is expected to occur at the end
            # If past the last timepoint, continue with the last parameter
            # value.
            previous_q_ = timecourse_q_[row_index - 1][1]
            sx_ += previous_q_ * (t - t_start)
            break
        t_end = timecourse_q_[row_index+1, 0]
        if t <= t_end:
            sx_ += q_ * (t - t_start)
            break
        sx_ += q_ * (t_end - t_start)

    return sx_


#@pytest.fixture
def yaml2sbml_model_string():
    """Defines the model used by the test."""
    return f"""
odes:
    - stateId: x_
      rightHandSide: p_ * q_
      initialValue: 0

parameters:
    - parameterId: p_
      nominalValue: {p_}
      lowerBound: 0.01
      upperBound: 100
      parameterScale: lin
      estimate: 1

    - parameterId: q_
      nominalValue: 1
      parameterScale: lin
      estimate: 0

observables:
    - observableId: observable_x_
      observableFormula: x_
      noiseFormula: 1

    - observableId: observable_q_
      observableFormula: q_
      noiseFormula: 1

conditions:
    - conditionId: condition1
    - conditionId: timecourse1
"""


#@pytest.fixture
def petab_yaml_path():
    petab_path = Path('output') / TEST_ID
    petab_path.mkdir(parents=True, exist_ok=True)

    petab_yaml_filename = 'petab.yaml'
    measurement_filename = 'measurements.tsv'

    with NamedTemporaryFile('w') as yaml2sbml_file:
        yaml2sbml_file.write(yaml2sbml_model_string())
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
