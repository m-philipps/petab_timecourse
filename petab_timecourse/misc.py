from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import petab
from petab.C import (
    OBSERVABLE_ID,
    MEASUREMENT,
    SIMULATION_CONDITION_ID,
    TIME,
    OBSERVABLE_FORMULA,
    NOISE_FORMULA,
)

from .C import (
    DUMMY_OBSERVABLE_ID,
    DUMMY_MEASUREMENT,
    DUMMY_NOISE,
    TYPE_PATH,
    TYPE_TIME,
    TIME_CONDITION_DELIMITER,
    PERIOD_DELIMITER,
    TIMECOURSE,
    # FIXME: Usage of this in this file overlaps with the `Condition` class in
    # `.petab`
    NON_COMPONENT_CONDITION_LABELS,
)


def get_path(path_like: TYPE_PATH) -> Path:
    return Path(path_like)


def times_to_durations(times: Iterable[float]) -> List[float]:
    return [
        times[i] - (times[i-1] if i != 0 else 0)
        for i, _ in enumerate(times)
    ]


def parse_timecourse_string(
    timecourse_string: str,
) -> List[Tuple[TYPE_TIME, List[str]]]:
    return [
        time_condition.split(TIME_CONDITION_DELIMITER)
        for time_condition in timecourse_string.split(PERIOD_DELIMITER)
    ]


def get_timecourse(petab_problem: petab.Problem, timecourse_id: str):
    return parse_timecourse_string(
        petab_problem.timecourse_df.loc[timecourse_id][TIMECOURSE],
    )


def parse_timecourse_string_as_lists(
        timecourse_string: str,
) -> Tuple[List[TYPE_TIME], List[str]]:
    # FIXME unify with above function
    result = parse_timecourse_string(timecourse_string)
    timepoints = []
    condition_ids = []
    for entry in result:
        timepoints.append(entry[0])
        condition_ids.append(entry[1])
    return timepoints, condition_ids


def subset_petab_problem(
    petab_problem: petab.Problem,
    timecourse_id: str,
) -> Sequence[petab.Problem]:
    petab_problem = deepcopy(petab_problem)
    # TODO allow no specification of timecourse if only one timecourse in
    # problem. TODO raise error if multiple timecourses but no timecourse ID
    # specified
    petab_problem.measurement_df = petab_problem.measurement_df[
        petab_problem.measurement_df[SIMULATION_CONDITION_ID] == timecourse_id
    ]
    timecourse = parse_timecourse_string(
        petab_problem.timecourse_df.loc[timecourse_id][TIMECOURSE],
    )
    # FIXME timepoints not necessarily float
    timecourse = [
        (float(_t), _id)
        for _t, _id in timecourse
    ]
    petab_problems = []
    for index, (timepoint, condition_id) in enumerate(timecourse):
        petab_problems.append(deepcopy(petab_problem))
        petab_problems[-1].condition_df.loc[timecourse_id] = \
            petab_problems[-1].condition_df.loc[condition_id]
        # Drop other conditions
        # FIXME test how this affects other conditions or how to combine normal conditions + timecourses...
        #       or only allow timecourses?
        petab_problems[-1].condition_df = (
            petab_problems[-1]
            .condition_df
            .loc[[timecourse_id]]
        )
        petab_problems[-1].measurement_df = \
            petab_problems[-1].measurement_df[
                petab_problems[-1].measurement_df[TIME].astype(float)
                >= timepoint
            ]
        if index < len(timecourse) - 1:
            next_timepoint = timecourse[index+1][0]
            petab_problems[-1].measurement_df = \
                petab_problems[-1].measurement_df[
                    petab_problems[-1].measurement_df[TIME].astype(float)
                    <= next_timepoint
                ]
        # Remove condition parameters from the parameters table.
        condition_components = [
            c
            for c in petab_problems[-1].condition_df.loc[timecourse_id].index
            if c not in NON_COMPONENT_CONDITION_LABELS
        ]
        petab_problems[-1].parameter_df.drop(
            condition_components,
            inplace=True,
            errors='ignore',  # only parameters that are in the table are dropped
        )

    return petab_problems
