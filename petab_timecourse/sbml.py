from typing import Any, Dict, Optional, Union

import libsbml
import pandas as pd
import petab
from slugify import slugify

#from .timecourse import Timecourse
from .petab import Condition
from .C import (
    TYPE_PATH,
    TIMECOURSE,
)
from .misc import parse_timecourse_string


def add_event(
        sbml_model: libsbml.Model,
        event_id: str,
        trigger_formula: str,
        event_assignments: Dict[str, Union[float, int, str]],
):
    """Add an event to an SBML model instance.

    The model is modified in-place.

    Parameters
    ----------
    sbml_model:
        The SBML model instance.
    event_id:
        The ID of the event.
    trigger_formula:
        The formula that describes when the event is triggered.
    event_assignments:
        A dictionary of assignments that occur when the event is triggered.
        Each key is an assignment target, and the corresponding value is value
        that the assignment target takes when the event is triggered.
    """
    event = sbml_model.createEvent()
    event.setId(event_id)
    event.setUseValuesFromTriggerTime(True)

    trigger = event.createTrigger()
    trigger.setInitialValue(True)
    trigger.setPersistent(True)
    trigger_math = libsbml.parseL3Formula(trigger_formula)
    trigger.setMath(trigger_math)

    for variable, event_assignment_formula in event_assignments.items():
        event_assignment = event.createEventAssignment()
        event_assignment.setVariable(variable)
        event_assignment_math = \
            libsbml.parseL3Formula(str(event_assignment_formula))
        event_assignment.setMath(event_assignment_math)


def add_timecourse_as_events(
        petab_problem: petab.Problem,
        #sbml_path: TYPE_PATH,
        timecourse_id: str = None,
        output_path: Optional[TYPE_PATH] = None,
):
    #sbml_path = str(sbml_path)
    #if output_path is None:
    #    output_path = sbml_path
    #output_path = str(output_path)

    if timecourse_id is None:
        try:
            timecourse_id = one(petab_problem.timecourse_df.index)
        except ValueError:
            raise ValueError(
                'A timecourse ID must be specified if there are multiple '
                'timecourses in the PEtab problem timecourse table.'
            )

    #sbml_document = libsbml.SBMLReader().readSBML(sbml_path)
    sbml_model = petab_problem.sbml_document.getModel()
    #if sbml_model is None:
    #    raise ValueError(
    #        'An SBML model could not be reproduced from the SBML file.'
    #    )

    timecourse = parse_timecourse_string(
        petab_problem.timecourse_df.loc[timecourse_id][TIMECOURSE],
    )

    for time, condition_id in timecourse:
        add_event(
            sbml_model=sbml_model,
            event_id=get_slug(time),
            trigger_formula=f'time >= {time}',
            event_assignments=Condition(
                petab_problem.condition_df.loc[condition_id],
            ),
        )

    if output_path is not None:
        libsbml.writeSBMLToFile(sbml_document, str(output_path))

    return sbml_model


def get_slug(
        input: Any,
        *args,
        **kwargs,
):
    """Get a human-readable string representation of the input.

    A description of what a slug is available at Wikipedia:
    https://en.wikipedia.org/wiki/Clean_URL#Slug

    The package `python-slugify` is used to generate the slug:
    https://github.com/un33k/python-slugify

    Parameters
    ----------
    input:
        The slug will be generated to represent this input.
    *args:
        Passed on to the `slugify` call (from the `python-slugify` package).
    **kwargs:
        Passed on to the `slugify` call (from the `python-slugify` package).
    """
    return slugify(str(input), *args, separator='_', **kwargs)
