from __future__ import annotations

from typing import Optional, Tuple, cast

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numerical.typedefs import ndarray

_Assign_gradinfo_t = type(None)


class Assign(Stage[_Assign_gradinfo_t]):
    lb: float
    ub: float

    def __init__(
        self,
        name: str,
        data_out_name: str,
        output: int,
        lb: float,
        ub: float,
    ) -> None:
        super().__init__((name,), (), (data_out_name,), (), (output,), ())
        self.lb = float(lb)
        self.ub = float(ub)

    def _eval(
        self, coeff: ndarray, x: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Assign_gradinfo_t]]:
        (length, _) = x.shape
        return numpy.full((length, 1), coeff), None

    def _grad(
        self, _: ndarray, output: _Assign_gradinfo_t, dL_dR: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        (length, _) = dL_dR.shape
        return numpy.empty((length, 0)), cast(ndarray, numpy.sum(dL_dR, axis=0))

    def get_constraints(self) -> Constraints:
        A = numpy.empty((0, 1))
        b = numpy.empty((0,))
        lb = numpy.array([self.lb])
        ub = numpy.array([self.ub])
        return Constraints(A, b, lb, ub)
