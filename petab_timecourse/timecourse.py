# TODO rename times to timepoints?
# TODO initial assignments in SBML for states are not supported. Should be
#      automatically detected and remove, or issue a warning?

from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Union

import amici
import numpy as np
import pandas as pd
from petab.C import (
    CONDITION_ID,
    CONDITION_NAME,
)

from .petab import Condition
from .C import (
    NON_COMPONENT_CONDITION_LABELS,
    TYPE_CONDITION_VALUE,
    TYPE_TIME,
    TIMECOURSE_ID,
    TIMECOURSE,
#    DEFAULT,
#    END,
#    START,
#    VALUE,
)
from .misc import parse_timecourse_string_as_lists
#from .sbml import (
#    add_event_to_sbml_model,
#)

#from .format_parameterwise import (  # FIXME refactor
#    Administration,
#    Regimen,
#)


class Timecourse():
    def __init__(
            self,
            row: pd.Series,
            #id: str,
            #timepoints: Iterable[TYPE_TIME],
            #conditions: Iterable[Condition],
            #regimens: Iterable[Regimen],
    ):
        # TODO change to init with timecourse_df
        #self.regimens = regimens
        # TODO sorting
        #self.id = id
        #self.timepoints = list(timepoints)
        #self.conditions = list(conditions)
        self.id = row.name

        #self.timecourse = parse_timecourse_string(row[TIMECOURSE])
        self.timepoints, self.condition_ids = parse_timecourse_string_as_lists(
            row[TIMECOURSE]
        )

        # FIXME argsort timepoints and condition IDs here or in `misc.parse...`
        #       or raise error if not ordered chronologically in timecourse
        #       string.

        
        #super().__init__(zip(*parse_timecourse_string(row[TIMECOURSE])))

    @staticmethod
    def from_df(
            timecourse_df: pd.DataFrame,
            timecourse_id: str,
    ) -> 'Timecourse':
        return Timecourse(timecourse_df.loc[timecourse_id])

    #@property
    #def times(self):
    #    return sorted({
    #        time
    #        for regimen in self.regimens
    #        for time in regimen.times
    #    })

    # FIXME: requires a condition DF
    #@property
    #def components(self):
    #    return set(chain(*[
    #        condition.components
    #        for condition in self.conditions
    #    ]))
    #    #return {*condition.components for condition in self.conditions}

    # FIXME: requires a condition DF
    #def componentwise(self):
    #    components = {}
    #    for timepoint, condition in self.items():
    #        for component_id, component_value in condition.items():
    #            components[component_id] = {
    #                **components.get(component_id, {}),
    #                **{timepoint: component_value}
    #            }
    #    return components



    #def values(self, time):
    #    #return {regimen.target: 0 for regimen in self.regimens}
    #    return {
    #        condition.target: regimen.value(time)
    #        for regimen in self.regimens
    #    }

    #def __iter__

    #def as_conditions_with_times(self):
    #    conditions = {}
    #    for time in self.times:
    #        conditions[time] = self.values(time)
    #    return conditions
