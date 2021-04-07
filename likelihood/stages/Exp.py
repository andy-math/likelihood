from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Stage
from numerical.typedefs import ndarray

_Exp_gradinfo_t = ndarray


class Exp(Stage[_Exp_gradinfo_t]):
    def __init__(self, input: int, output: int) -> None:
        super().__init__([], (input,), (output,))

    def _eval(
        self, _: ndarray, x: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Exp_gradinfo_t]]:
        output = numpy.exp(x)
        if not grad:
            return output, None
        return output, output

    def _grad(
        self, _: ndarray, output: _Exp_gradinfo_t, dL_dR: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        return dL_dR * output, numpy.ndarray((0,))

    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return A, b, lb, ub
