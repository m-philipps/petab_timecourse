"""Constants."""
from pathlib import Path
from typing import Union

import petab.C


TIMECOURSE = 'timecourse'
TIMECOURSE_ID = 'timecourseId'
TIME_CONDITION_DELIMITER = ':'
PERIOD_DELIMITER = ';'
TIMECOURSE_NAME = 'timecourseName'
PERIOD = 'period'
PERIODS = 'periods'


# Parameterwise file values.
START = 'start'
END = 'end'
VALUE = 'value'
DEFAULT = 'default'

TYPE_PATH = Union[str, Path]
# A timecourse timepoint may be any of the following.
# - The time (a number).
#   - Negative values indicate pre-simulation.
# - A parameter ID (a string).
# - A special time (a string):
#   - `'preequilibration'` or `'-inf'`
TYPE_TIME = Union[float, int, str]

# FIXME move to petab.C
CONDITION = 'condition'
TYPE_CONDITION_VALUE = Union[float, int, str]

NON_COMPONENT_CONDITION_LABELS = [petab.C.CONDITION_ID, petab.C.CONDITION_NAME]

DUMMY_OBSERVABLE_ID = 'timecourse_dummy_observable_id'
DUMMY_MEASUREMENT = 1
DUMMY_NOISE = 1

ESTIMATE = 'estimate'

FINAL_STATES = 'final_states'
RESULTS = 'results'
