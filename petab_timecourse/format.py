from copy import deepcopy
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import petab
from petab.C import (
    CONDITION_ID,
    CONDITION_NAME,
)

from .timecourse import (
    Timecourse,
    Condition,
)
from .petab import Condition
from .C import (
    CONDITION,
    TIMECOURSE,
    TIMECOURSE_ID,
    TIMECOURSE_NAME,
    TYPE_PATH,
    TIME_CONDITION_DELIMITER,
    PERIOD_DELIMITER,
)
from .format_parameterwise import (  # FIXME: refactor to remove dependency
    Regimen,
    Regimens,
)
# FIXME remove imports from format.py
# FIXME remove references to this
from .timecourse import get_timecourse_df


def deduplicate_conditions(conditions: Sequence[Condition]) -> Sequence[str]:
    # Sort all conditions by column name, such that after conversion to string
    # they can be compared.
    sorted_conditions = [
        {
            component: condition[component]
            for component in sorted(condition)
        }
        for condition in conditions
    ]

    # Identify earliest index that is a duplicate of a later index in the list
    # of condition dictionaries.
    _, unique_indices, duplicated_indices = np.unique(
        np.array(sorted_conditions).astype(str),
        return_index=True,
        return_inverse=True,
    )

    unique_sequence = [unique_indices[i] for i in duplicated_indices]
    unique_conditions = {
        conditions[i].id: conditions[i]
        for i in unique_sequence
    }
    condition_sequence = [conditions[i].id for i in unique_sequence]
    return unique_conditions, condition_sequence


def import_directory_of_componentwise_files(
        directory: TYPE_PATH,
        timecourse_id: str = None,
) -> Tuple[Timecourse, pd.DataFrame]:
    directory = Path(directory)
    if timecourse_id is None:
        timecourse_id = f'timecourse_{directory.parts[-1]}'

    regimens = Regimens({
        Regimen.from_path(path)
        for path in Path(directory).iterdir()
    })
    conditions_with_times = regimens.as_conditions()
    for index, (time, condition) in enumerate(conditions_with_times.items()):
        conditions_with_times[time] = Condition(pd.Series(
            data=condition,
            name=f'{timecourse_id}_condition_{index}',
        ))
    unique_conditions, condition_sequence = \
        deduplicate_conditions(list(conditions_with_times.values()))
    timecourse_df = pd.DataFrame(data={
        TIMECOURSE_ID: [f'{timecourse_id}'],
        TIMECOURSE: [PERIOD_DELIMITER.join([
            f'{timepoint}{TIME_CONDITION_DELIMITER}{condition_id}'
            for timepoint, condition_id in \
                zip(conditions_with_times, condition_sequence)
        ])],
    })
    timecourse_df = get_timecourse_df(timecourse_df)
    if len(timecourse_df) != 1:
        raise ValueError(
            'Something went wrong with importing the componentwise timecourse.'
            'Multiple timecourses were created.'
        )

    # TODO duplicated from "to_petab_files"...
    condition_df = pd.DataFrame(data=[
        {
            **{
                CONDITION_ID: condition.condition_id,
                CONDITION_NAME: condition.name,
            },
            **dict(condition),
        }
        for condition in unique_conditions.values()
    ])
    if set(condition_df[CONDITION_NAME]) is None:
        condition_df.drop(CONDITION_NAME)
    condition_df = petab.get_condition_df(condition_df)

    return timecourse_df, condition_df


def from_petab(yaml_location: TYPE_PATH) -> Timecourse:
    # TODO
    yaml_location = Path(yaml_location)
    pass


def to_petab_dataframes(timecourse: Timecourse) -> Dict[str, pd.DataFrame]:
    """Convert a timecourse to PEtab dataframes.

    Parameters
    ----------
    timecourse:
        The timecourse to convert.

    Returns
    -------
    A dictionary, where the keys are `'.C.CONDITION'` and `.C.TIMECOURSE`,
    and the values are the corresponding dataframes.
    """
    condition_df = pd.DataFrame(data=[
        dict(
            CONDITION_ID=condition.condition_id,
            CONDITION_NAME=condition.name,
            **dict(condition),
        )
        for condition in timecourse.values()
    ])
    if set(condition_df[CONDITION_NAME]) is None:
        condition_df.drop(CONDITION_NAME)
    condition_df = petab.get_condition_df(condition_df)

    timecourse_df = pd.DataFrame(data=[
        dict(
            TIMECOURSE_ID=timecourse.timecourse_id,
            TIMECOURSE=TIMECOURSE_ITEM_DELIMITER.join([
                f'{timepoint}{TIME_CONDITION_DELIMITER}{condition.condition_id}'
                for timepoint, condition in timecourse.items()
            ]),
        )
    ])
    if set(timecourse_df[TIMECOURSE_NAME]) is None:
        timecourse_df.drop(TIMECOURSE_NAME)
    timecourse_df = get_timecourse_df(timecourse_df)

    return {
        CONDITION: condition_df,
        TIMECOURSE: timecourse_df,
    }


def write_timecourse_df(
        timecourse_df,
        filename: str,
):
    timecourse_df.to_csv(filename, sep='\t', index=True)


def to_petab_files(
        timecourse: Timecourse,
        condition_location: TYPE_PATH,
        timecourse_location: TYPE_PATH,
) -> pd.DataFrame:
    dfs = to_petab_dataframes(timecourse)
    petab.write_condition_df(dfs[CONDITION], str(condition_location))
    write_timecourse_df(dfs[TIMECOURSE], str(timecourse_location))
