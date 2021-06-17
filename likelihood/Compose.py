from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray

from likelihood.stages.abc.Stage import Stage


class Compose:
    def eval(
        self,
        stages: List[Stage[Any]],
        coeff: ndarray,
        input: ndarray,
        *,
        grad: bool,
        debug: bool
    ) -> Tuple[ndarray, Optional[List[Any]]]:
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
        self,
        stages: List[Stage[Any]],
        coeff: ndarray,
        gradinfo: List[Any],
        dL_do: ndarray,
        *,
        debug: bool
    ) -> Tuple[ndarray, ndarray]:
        dL_dc = numpy.zeros(coeff.shape)
        for s, g in zip(stages[::-1], gradinfo[::-1]):
            assert s.coeff_index is not None
            dL_do, _dL_dc = s.grad(coeff[s.coeff_index], g, dL_do, debug=debug)
            for i, ci in enumerate(s.coeff_index):
                dL_dc[ci] += _dL_dc[i]
        return dL_do, dL_dc
