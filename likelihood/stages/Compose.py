from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy
from likelihood.stages.abc.Stage import Stage
from numerical.typedefs import ndarray


def _make_names(*stages: Stage[Any]) -> Tuple[List[str], List[int]]:
    names: List[str] = []
    packing: List[int] = []
    for s in stages:
        names.extend(s.names)
        packing.append(len(s.names))
    assert len(set(names)) == len(names)
    return names, numpy.cumsum(packing)  # type: ignore


_Compose_gradinfo_t = List[Any]


class Compose(Stage[_Compose_gradinfo_t]):
    len_coeff: int
    packing: List[int]
    stages: List[Stage[Any]]

    def __init__(
        self, stages: List[Stage[Any]], input: Sequence[int], output: Sequence[int]
    ) -> None:
        names, packing = _make_names(*stages)
        super().__init__(names, input, output)
        assert len(input) == len(output)
        for s in stages:
            assert max(s._input_idx) < len(input)
            assert max(s._output_idx) < len(output)
        self.len_coeff = packing[-1]
        self.packing = packing[:-1]
        self.stages = stages

    def _unpack(self, coeff: ndarray) -> List[ndarray]:
        return numpy.split(coeff, self.packing)

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Compose_gradinfo_t]]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        output: ndarray = input
        gradinfo: List[Optional[Any]] = []
        for s, c in zip(stages, coeffs):
            output, g = s.eval(c, output, grad=grad)
            gradinfo.append(g)
        if not grad:
            return output, None
        return output, gradinfo

    def _grad(
        self, coeff: ndarray, gradinfo: _Compose_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        dL_dc: List[ndarray] = []
        for s, c, g in zip(stages[::-1], coeffs[::-1], gradinfo[::-1]):
            dL_do, _dL_dc = s.grad(c, g, dL_do)
            dL_dc.append(_dL_dc)
        return dL_do, numpy.concatenate(dL_dc[::-1])

    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        from scipy.linalg import block_diag  # type: ignore

        A, b, lb, ub = zip(*(s.get_constraint() for s in self.stages))
        return (
            block_diag(*A),
            numpy.concatenate(b),
            numpy.concatenate(lb),
            numpy.concatenate(ub),
        )
