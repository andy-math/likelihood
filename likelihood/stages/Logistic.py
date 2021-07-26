from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from overloads.typedefs import ndarray

_Logistic_gradinfo_t = ndarray


class Logistic(Stage[_Logistic_gradinfo_t]):
    def __init__(
        self, data_in_names: Tuple[str, ...], data_out_names: Tuple[str, ...]
    ) -> None:
        assert len(data_in_names) == len(data_out_names)
        super().__init__((), data_in_names, data_out_names, ())

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

    def get_constraints(self) -> Constraints:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return Constraints(A, b, lb, ub)
