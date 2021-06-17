from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import isunique

from likelihood.stages.abc.Stage import Constraints, Stage


def _make_names(*stages: Stage[Any]) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    coeff_names: List[str] = []
    packing: List[int] = []
    for s in stages:
        coeff_names.extend(s.coeff_names)
        packing.append(len(s.coeff_names))
    assert isunique(coeff_names)
    return tuple(coeff_names), tuple(numpy.cumsum(packing).tolist())


_Compose_gradinfo_t = List[Any]


class Compose:
    names: Tuple[str, ...]
    len_coeff: int
    packing: Tuple[int, ...]
    stages: List[Stage[Any]]

    def __init__(self, stages: List[Stage[Any]], nvars: int) -> None:
        names, packing = _make_names(*stages)
        self.names = names
        for s in stages:
            assert not len(s.data_in_index) or max(s.data_in_index) < nvars
            assert not len(s.data_out_index) or max(s.data_out_index) < nvars
        self.len_coeff = packing[-1]
        self.packing = packing[:-1]
        self.stages = stages

    def _unpack(self, coeff: ndarray) -> List[ndarray]:
        return numpy.split(coeff, self.packing)  # type: ignore

    def eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Compose_gradinfo_t]]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        output: ndarray = input
        gradinfo: List[Optional[Any]] = []
        for s, c in zip(stages, coeffs):
            output, g = s.eval(c, output, grad=grad, debug=debug)
            gradinfo.append(g)
        if not grad:
            return output, None
        return output, gradinfo

    def grad(
        self,
        coeff: ndarray,
        gradinfo: _Compose_gradinfo_t,
        dL_do: ndarray,
        *,
        debug: bool
    ) -> Tuple[ndarray, ndarray]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        dL_dc: List[ndarray] = []
        for s, c, g in zip(stages[::-1], coeffs[::-1], gradinfo[::-1]):
            dL_do, _dL_dc = s.grad(c, g, dL_do, debug=debug)
            dL_dc.append(_dL_dc)
        return dL_do, numpy.concatenate(dL_dc[::-1])

    def get_constraint(self) -> Constraints:
        from scipy.linalg import block_diag  # type: ignore

        A, b, lb, ub = zip(*(s.get_constraints() for s in self.stages))
        return Constraints(
            block_diag(*A),
            numpy.concatenate(b),
            numpy.concatenate(lb),
            numpy.concatenate(ub),
        )
