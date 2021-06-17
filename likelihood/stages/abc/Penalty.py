from abc import ABCMeta
from typing import Tuple, TypeVar

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage

_Penalty_gradinfo_t = TypeVar("_Penalty_gradinfo_t")


class Penalty(Stage[_Penalty_gradinfo_t], metaclass=ABCMeta):
    coeff_names: Tuple[str, ...]

    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        data_in_names: Tuple[str, ...],
        data_out_names: Tuple[str, ...],
        input: Tuple[int, ...],
        output: Tuple[int, ...],
    ) -> None:
        super().__init__((), data_in_names, data_out_names, input, output)
        self.coeff_names = coeff_names

    def get_constraints(self) -> Constraints:
        return Constraints(
            numpy.empty((0, len(self.coeff_names))),
            numpy.empty((0,)),
            numpy.full((len(self.coeff_names),), -numpy.inf),
            numpy.full((len(self.coeff_names),), numpy.inf),
        )
