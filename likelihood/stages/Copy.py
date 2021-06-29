from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numpy import ndarray

_Copy_gradinfo_t = type(None)


class Copy(Stage[_Copy_gradinfo_t]):
    def __init__(
        self,
        data_in_names: Tuple[str, ...],
        data_out_names: Tuple[str, ...],
    ) -> None:
        assert len(data_in_names) == len(data_out_names)
        super().__init__((), data_in_names, data_out_names, ())

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
