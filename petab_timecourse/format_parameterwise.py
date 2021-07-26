from pathlib import Path
from typing import Iterable

import pandas as pd
from more_itertools import one

from .C import (
    DEFAULT,
    END,
    START,
    VALUE,

    ESTIMATE,
)


class Administration():
    # TODO as described elsewhere, values are currently assumed to be float.
    #      May need to be generalised to strings later.
    def __init__(
            self,
            start: float,
            end: float,
            value: float,
            estimate: bool = False,
    ):
        self._start = start
        self._end = end
        self._value = value
        self._estimate = estimate

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def value(self):
        return self._value

    @property
    def estimate(self):
        return self._estimate


class Regimen():
    # TODO change file_ to df in __init__?
    # TODO currently designed as read-only-after-initialization
    def __init__(self, target: str, df: pd.DataFrame):
        self._target = target
        self._administrations = set()
        self._times = set()
        self._default = None
        for _, row in df.iterrows():
            # TODO assumes all values are float.. could replace with
            #      try-except; however, times are sorted later, so should be
            #      float anyway... will need to change for optimal control
            estimate = False
            try:
                value = float(row[VALUE])
            except ValueError:
                if row[VALUE] == ESTIMATE:
                    value = None
                    estimate = True
                else:
                    raise
            if row[START] == DEFAULT:
                self._default = value
                if estimate == True:
                    raise NotImplementedError(
                        'The default value cannot be estimated.'
                    )
                continue
            if row[VALUE] == ESTIMATE:
                value = None
            start = float(row[START])
            end = float(row[END])
            self._administrations.add(Administration(
                start=start,
                end=end,
                value=value,
                estimate=estimate,
            ))
            self._times |= {start, end}
        assert '_default' in dir(self)
        assert self.default is not None
        assert not pd.isna(self.default)
        assert '_times' in dir(self)
        self._times = sorted(self._times)

    @staticmethod
    def from_path(path: Path) -> 'Regimen':
        """TODO Assumes the name of the target is the stem of the path."""
        with open(path, 'r') as f:
            df = pd.read_csv(f, sep='\t')
        return Regimen(path.stem, df)

    @property
    def target(self):
        return self._target

    @property
    def default(self):
        return self._default

    @property
    def times(self):
        return self._times

    def value(self, time):
        values = set()
        for administration in self._administrations:
            # TODO assumes >= start, < end
            if time >= administration.start and time < administration.end:
                values.add(administration.value)
        # Currently assumed that at most one administration will match.
        return one(values) if values else self.default


class Regimens():
    def __init__(self, regimens: Iterable[Regimen]):
        self.regimens = regimens

    @staticmethod
    def from_paths(paths: Iterable[Path]) -> 'Regimens':
        return Regimens({
            Regimen.from_path(path)
            for path in paths
        })

    @property
    def times(self):
        return sorted({
            time
            for regimen in self.regimens
            for time in regimen.times
        })

    @property
    def targets(self):
        return {regimen.target for regimen in self.regimens}

    def values(self, time):
        #return {regimen.target: 0 for regimen in self.regimens}
        return {
            regimen.target: regimen.value(time)
            for regimen in self.regimens
        }

    #def as_single_piecewises(self) -> Dict[str, str]:
    #    return {
    #        regimen.target: regimen.as_single_piecewise()
    #        for regimen in self.regimens
    #    }

    #def as_additive_piecewises(self) -> Dict[str, str]:
    #    return {
    #        regimen.target: regimen.as_additive_piecewise()
    #        for regimen in self.regimens
    #    }

    def as_conditions(self):
        conditions = {}
        for time in self.times:
            conditions[time] = self.values(time)
        return conditions

    #def add_as_events_to_sbml_file(
    #        self,
    #        sbml_path: Union[str, Path],
    #        output_path: Optional[Union[str, Path]] = None,
    #):
    #    sbml_path = str(sbml_path)
    #    if output_path is None:
    #        output_path = sbml_path
    #    output_path = str(output_path)

    #    sbml_document = libsbml.SBMLReader().readSBML(sbml_path)
    #    sbml_model = sbml_document.getModel()
    #    if sbml_model is None:
    #        raise ValueError(
    #            'An SBML model could not be reproduced from the SBML file.'
    #        )

    #    for time, condition in self.as_conditions().items():
    #        add_event_to_sbml_model(
    #            sbml_model=sbml_model,
    #            event_id=get_slug(time),
    #            trigger_formula=f'time >= {time}',
    #            event_assignments=condition,
    #        )

    #    libsbml.writeSBMLToFile(sbml_document, output_path)
