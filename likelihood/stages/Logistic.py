from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numerical.typedefs import ndarray

_Logistic_gradinfo_t = ndarray


class Logistic(Stage[_Logistic_gradinfo_t]):
    def __init__(self, input: Tuple[int, ...], output: Tuple[int, ...]) -> None:
        assert len(input) == len(output)
        super().__init__((), input, output)

    def _eval(
        self, _: ndarray, x: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Logistic_gradinfo_t]]:
        """
        logistic(x) = exp(x)/(1+exp(x))
                    = 1/(exp(-x)+1)  # exp overflow safety
        """
        output: ndarray = 1.0 / (numpy.exp(-x) + 1.0)
        if not grad:
            return output, None
        return output, output

    def _grad(
        self, var: ndarray, output: _Logistic_gradinfo_t, dL_dR: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        """
        dL_dx = -1*1/(exp(-x)+1)^2 * exp(-x) * -1
              = 1/(exp(-x)+1) * exp(-x)/(exp(-x)+1)
              = logistic * (1-logistic)
        """
        return dL_dR * (output * (1.0 - output)), numpy.empty((0,))

    def get_constraint(self) -> Constraints:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return Constraints(A, b, lb, ub)
