from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numerical.typedefs import ndarray

_Log_gradinfo_t = ndarray


class Log(Stage[_Log_gradinfo_t]):
    def __init__(self, input: int, output: int) -> None:
        super().__init__((), (input,), (output,))

    def _eval(
        self, _: ndarray, x: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Log_gradinfo_t]]:
        output = numpy.log(x)
        if not grad:
            return output, None
        return output, x

    def _grad(
        self, _: ndarray, x: _Log_gradinfo_t, dL_dR: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        return dL_dR / x, numpy.ndarray((0,))

    def get_constraints(self) -> Constraints:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return Constraints(A, b, lb, ub)
