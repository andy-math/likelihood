from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numerical.typedefs import ndarray

_Residual_gradinfo_t = type(None)


class Residual(Stage[_Residual_gradinfo_t]):
    def __init__(
        self,
        data_in_names: Tuple[str, str],
        data_out_name: str,
    ) -> None:
        super().__init__((), data_in_names, (data_out_name,), ())

    def _eval(
        self, _: ndarray, x_mu: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Residual_gradinfo_t]]:
        return x_mu[:, [0]] - x_mu[:, [1]], None

    def _grad(
        self, var: ndarray, _: _Residual_gradinfo_t, dL_dR: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        return dL_dR * numpy.array([[1.0, -1.0]]), numpy.ndarray((0,))

    def get_constraints(self) -> Constraints:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return Constraints(A, b, lb, ub)
