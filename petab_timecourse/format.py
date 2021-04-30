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
    TIME_CONDITION_DELIMETER,
    TIMECOURSE_ITEM_DELIMETER,
)
#from .misc import get_path
from .format_parameterwise import (  # FIXME: refactor to remove dependency
    Regimen,
    Regimens,
)


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
) -> Tuple[Timecourse, pd.DataFrame]:
    regimens = Regimens({
        Regimen.from_path(path)
        #for path in Path(directory).iterdir()
        for path in Path(directory).rglob('*.tsv')  # FIXME remove and replace with above line
    })
    #conditions_with_times = regimens.as_conditions()  # FIXME uncoment and remove below quickfixes
    #conditions_with_times = {time: value for time, value in regimens.as_conditions().items() if time != 0}
    conditions_with_times = {time + 0.1: value for time, value in regimens.as_conditions().items()}
    for index, (time, condition) in enumerate(conditions_with_times.items()):
        conditions_with_times[time] = Condition(pd.Series(
            data=condition,
            name=f'timecourse_{directory.parts[-1]}_condition_{index}',
        ))
    unique_conditions, condition_sequence = \
        deduplicate_conditions(list(conditions_with_times.values()))
    timecourse_df = pd.DataFrame(data={
        TIMECOURSE_ID: [f'timecourse_{directory.parts[-1]}'],
        TIMECOURSE: [TIMECOURSE_ITEM_DELIMETER.join([
            f'{timepoint}{TIME_CONDITION_DELIMETER}{condition_id}'
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
    #timecourse = Timecourse(timecourse_df.iloc[1])

    # TODO duplicated from "to_petab_files"...
    condition_df = pd.DataFrame(data=[
        {
            **{
                CONDITION_ID: condition.id,
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

    #return Timecourse(timecourse_df.iloc[1]), condition_df
    #for timecourse_id, timecourse_row in in timecourse_df.iterrows():
    #return Timecourse
    #breakpoint()
    #return Timecourse(
    #    id='from_componentwise',
    #    timepoints=conditions_with_times.keys(),
    #    conditions=[
    #        Condition(pd.Series(data={
    #            CONDITION_ID: directory.parts[-1],
    #            #**dict(zip(regimens.targets, condition)),
    #            **condition,
    #        }))
    #        for condition in conditions_with_times.values()
    #    ],
    #)



def from_petab(yaml_location: TYPE_PATH) -> Timecourse:
    # TODO
    yaml_location = Path(yaml_location)
    pass


def get_timecourse_df(
        timecourse_file: Union[str, pd.DataFrame, None]
) -> pd.DataFrame:
    """Read the provided condition file into a ``pandas.Dataframe``
    Conditions are rows, parameters are columns, conditionId is index.
    Arguments:
        condition_file: File name of PEtab condition file or pandas.Dataframe
    """
    if timecourse_file is None:
        return timecourse_file

    if isinstance(timecourse_file, str) or isinstance(timecourse_file, Path):
        timecourse_file = pd.read_csv(
            timecourse_file,
            sep='\t',
            float_precision='round_trip',
        )

    petab.lint.assert_no_leading_trailing_whitespace(
        timecourse_file.columns.values, "timecourse"
    )

    if not isinstance(timecourse_file.index, pd.RangeIndex):
        timecourse_file.reset_index(inplace=True)

    try:
        timecourse_file.set_index([TIMECOURSE_ID], inplace=True)
    except KeyError:
        raise KeyError(
            f'Timecourse table missing mandatory field {TIMECOURSE_ID}.')

    return timecourse_file


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
            CONDITION_ID=condition.id,
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
            TIMECOURSE_ID=timecourse.id,
            TIMECOURSE=TIMECOURSE_ITEM_DELIMETER.join([
                f'{timepoint}{TIME_CONDITION_DELIMETER}{condition.id}'
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
