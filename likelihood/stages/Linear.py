from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numerical.typedefs import ndarray

_Linear_gradinfo_t = ndarray


class Linear(Stage[_Linear_gradinfo_t]):
    def __init__(
        self,
        names: Tuple[str, ...],
        data_in_names: Tuple[str, ...],
        data_out_name: str,
    ) -> None:
        super().__init__(names, data_in_names, (data_out_name,), ())

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Linear_gradinfo_t]]:
        output = (input @ coeff).reshape((-1, 1))
        if not grad:
            return output, None
        return output, input

    def _grad(
        self, coeff: ndarray, input: _Linear_gradinfo_t, dL_do: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        return dL_do * coeff, dL_do.flatten() @ input

    def get_constraints(self) -> Constraints:
        A = numpy.empty((0, len(self.coeff_names)))
        b = numpy.empty((0,))
        lb = numpy.full((len(self.coeff_names),), -numpy.inf)
        ub = numpy.full((len(self.coeff_names),), numpy.inf)
        return Constraints(A, b, lb, ub)
