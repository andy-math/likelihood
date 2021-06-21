from abc import ABCMeta
from typing import Tuple, TypeVar

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage

_Penalty_gradinfo_t = TypeVar("_Penalty_gradinfo_t")


class Penalty(Stage[_Penalty_gradinfo_t], metaclass=ABCMeta):
    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        data_in_names: Tuple[str, ...],
        data_out_names: Tuple[str, ...],
    ) -> None:
        super().__init__(coeff_names, data_in_names, data_out_names, ())

    def get_constraints(self) -> Constraints:
        return Constraints(
            numpy.empty((0, len(self.coeff_names))),
            numpy.empty((0,)),
            numpy.full((len(self.coeff_names),), -numpy.inf),
            numpy.full((len(self.coeff_names),), numpy.inf),
        )
