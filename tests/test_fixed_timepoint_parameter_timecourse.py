from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence, Tuple
from warnings import warn

import amici
from amici.petab import import_petab_problem
from amici.petab import simulate_petab
import numpy as np
import pandas as pd
import petab
import pytest
import yaml2sbml


from fixture_fixed_timepoint_parameter_timecourse import (
    OUTPUT_DENSITY,
    petab_yaml_path,
    timecourse_q_,

    get_analytical_x_,
    get_analytical_sx_
)


def test_model():
    petab_problem = petab.Problem.from_yaml(str(petab_yaml_path()))
    model = import_petab_problem(petab_problem, compile_=None)  # FIXME compile_ True
    solver = model.getSolver()
    solver.setSensitivityOrder(1)
    solver.setSensitivityMethod(1)
    
    rdatas = []
    rdata = None
    for row_index, (t_start, q_) in enumerate(timecourse_q_):
        if q_ is None:
            break
        t_end = timecourse_q_[row_index+1, 0]
        model.setTimepoints(
            np.linspace(t_start, t_end, OUTPUT_DENSITY*(t_end-t_start) + 1)
        )
        #model.setFixedParameterById('q_', q_)
        model.setFixedParameterById('q_', q_)
    
        # Continue the next sub-simulation from the previous simulation.
        if rdatas:
            model.setT0(t_start)
            model.setInitialStates(rdatas[-1].x[-1])
            # 0th column is sensitivity of x_ w.r.t. p_
            # 1st column is sensitivity of x_ w.r.t. q_
            model.setInitialStateSensitivities(rdatas[-1].sx[-1][:, 0])
        rdatas.append(amici.runAmiciSimulation(model, solver))
    
    # Collect
    # FIXME magic number `0`
    x_ = list(chain.from_iterable([
        rdata.x[:, 0].flatten()
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
