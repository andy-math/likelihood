from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numerical.typedefs import ndarray

_Exp_gradinfo_t = ndarray


class Exp(Stage[_Exp_gradinfo_t]):
    def __init__(
        self,
        data_in_name: str,
        data_out_name: str,
    ) -> None:
        super().__init__((), (data_in_name,), (data_out_name,), ())

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

    def get_constraints(self) -> Constraints:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return Constraints(A, b, lb, ub)
