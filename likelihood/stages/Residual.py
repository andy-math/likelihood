from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Logpdf import Logpdf
from numerical.typedefs import ndarray

_Residual_gradinfo_t = type(None)


class Residual(Logpdf[_Residual_gradinfo_t]):
    def __init__(self, input: Tuple[int, int], output: int) -> None:
        super().__init__([], input, (output,))

    def _eval(
        self, _: ndarray, x_mu: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Residual_gradinfo_t]]:
        return x_mu[:, [0]] - x_mu[:, [1]], None  # type: ignore

    def _grad(
        self, var: ndarray, _: _Residual_gradinfo_t, dL_dR: ndarray
    ) -> Tuple[ndarray, ndarray]:
        return dL_dR * numpy.array([[1.0, -1.0]]), numpy.ndarray((0,))

    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return A, b, lb, ub
