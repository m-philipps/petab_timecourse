from typing import Any, Dict, Iterable, List, Optional, TextIO, Union

import pandas as pd
import petab
from petab.C import (
    CONDITION_ID,
    CONDITION_NAME,
    PARAMETER_SCALE,
)

from .C import (
    NON_COMPONENT_CONDITION_LABELS,
)

class Condition(dict):
    """A PEtab condition.

    Attributes
    ----------
    id:
        The condition ID.
    components:
        The model components that are altered by the condition.
    values:
        The values that the components take.
    """

    def __init__(
            self,
            row: pd.Series,
    ):
        """

        Parameters
        ----------
        row:
            A row from a PEtab conditions table.
        """
        components = {
            component: value
            for component, value in row.items()
            if component not in NON_COMPONENT_CONDITION_LABELS
        }
        super().__init__(components)
        self.id = row.name
        self.name = row.get(CONDITION_NAME, None)
        #self.condition_id = row[CONDITION_ID]
        #self.condition_name = row.get(CONDITION_NAME, None)

        #row_components_only = row.drop(labels=[CONDITION_ID, CONDITION_NAME])
        #self.components = row_components_only.components()
        #self.values = row_components_only.values()

    #@property
    #def id(self):
    #    return self.get(CONDITION_ID)

    #@property
    #def name(self):
    #    return self.get(CONDITION_NAME, None)

    @property
    def components(self):
        return list(self.keys())
        #return [
        #    component
        #    for component in self.index
        #    if component not in NON_COMPONENT_CONDITION_LABELS
        #]

    def as_series(self):
        return pd.Series(data={
            CONDITION_ID: self.id,
            CONDITION_NAME: self.name,
            **self.items(),
        })

    #@property
    #def component_values(self):
    #    return list(self.values())
    #    #return [
    #    #    value
    #    #    for component, value in self.items()
    #    #    if component not in NON_COMPONENT_CONDITION_LABELS
    #    #]
#
#
#def unscale_parameters(
#    scaled_parameters: Dict[str, float],
#    petab_problem: petab.Problem,
#):
#    scales = dict(petab_problem.parameter_df[PARAMETER_SCALE])
#
#    unscaled_parameters = {
#        parameter_id: petab.parameters.unscale(
#            parameter_value,
#            scales[parameter_id],
#        )
#        for parameter_id, parameter_value in scaled_parameters.items()
#    }
#
#    return unscaled_parameters
