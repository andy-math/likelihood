from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numerical.typedefs import ndarray

_Copy_gradinfo_t = type(None)


class Copy(Stage[_Copy_gradinfo_t]):
    def __init__(self, input: Tuple[int, ...], output: Tuple[int, ...]) -> None:
        assert len(input) == len(output)
        super().__init__((), input, output)

    def _eval(
        self, _: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Copy_gradinfo_t]]:
        return input, None

    def _grad(
        self, _: ndarray, __: _Copy_gradinfo_t, dL_dR: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        return dL_dR, numpy.ndarray((0,))

    def get_constraints(self) -> Constraints:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return Constraints(A, b, lb, ub)
