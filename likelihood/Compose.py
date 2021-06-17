from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray

from likelihood.stages.abc.Stage import Constraints, Stage


class Compose:
    stages: List[Stage[Any]]

    def __init__(self, stages: List[Stage[Any]], nvars: int) -> None:
        for s in stages:
            assert not len(s.data_in_index) or max(s.data_in_index) < nvars
            assert not len(s.data_out_index) or max(s.data_out_index) < nvars
        self.stages = stages

    def eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[List[Any]]]:
        stages = self.stages
        output: ndarray = input
        gradinfo: List[Optional[Any]] = []
        for s in stages:
            assert s.coeff_index is not None
            output, g = s.eval(coeff[s.coeff_index], output, grad=grad, debug=debug)
            gradinfo.append(g)
        if not grad:
            return output, None
        return output, gradinfo

    def grad(
        self, coeff: ndarray, gradinfo: List[Any], dL_do: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        stages = self.stages
        dL_dc: List[ndarray] = []
        for s, g in zip(stages[::-1], gradinfo[::-1]):
            assert s.coeff_index is not None
            dL_do, _dL_dc = s.grad(coeff[s.coeff_index], g, dL_do, debug=debug)
            dL_dc.append(_dL_dc)
        return dL_do, numpy.concatenate(dL_dc[::-1])

    def get_constraints(self) -> Constraints:
        from scipy.linalg import block_diag  # type: ignore

        A, b, lb, ub = zip(*(s.get_constraints() for s in self.stages))
        return Constraints(
            block_diag(*A),
            numpy.concatenate(b),
            numpy.concatenate(lb),
            numpy.concatenate(ub),
        )
