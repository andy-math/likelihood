from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from numerical.typedefs import ndarray


def _make_packing(idx: List[Sequence[int]]) -> List[int]:
    packing: List[int] = []
    for i in idx:
        packing.append(len(i))
    return numpy.cumsum(packing).tolist()[:-1]  # type: ignore


def _make_names(*stages: Stage[Any]) -> Tuple[List[str], List[int]]:
    names: List[str] = []
    packing: List[int] = []
    for s in stages:
        names.extend(s.names)
        packing.append(len(s.names))
    assert len(set(names)) == len(names)
    return names, numpy.cumsum(packing).tolist()


_Merge_gradinfo_t = List[Any]


class Merge(Stage[_Merge_gradinfo_t]):
    len_coeff: int
    packing: List[int]
    packing_input: List[int]
    packing_output: List[int]
    stages: List[Stage[Any]]

    def __init__(self, stages: List[Stage[Any]]) -> None:
        names, packing = _make_names(*stages)
        input: List[int] = []
        output: List[int] = []
        for s in stages:
            input.extend(s._input_idx)
            output.extend(s._output_idx)
        super().__init__(names, input, output)
        self.len_coeff = packing[-1]
        self.packing = packing[:-1]
        self.stages = stages
        self.packing_input = _make_packing([s._input_idx for s in stages])
        self.packing_output = _make_packing([s._output_idx for s in stages])

    def _unpack(self, coeff: ndarray) -> List[ndarray]:
        return numpy.split(coeff, self.packing)  # type: ignore

    def _unpack_input(self, input: ndarray) -> List[ndarray]:
        return numpy.split(input, self.packing_input, axis=1)  # type: ignore

    def _unpack_output(self, output: ndarray) -> List[ndarray]:
        return numpy.split(output, self.packing_input, axis=1)  # type: ignore

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Merge_gradinfo_t]]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        output: List[ndarray] = []
        gradinfo: List[Optional[Any]] = []
        for s, c, _input in zip(stages, coeffs, self._unpack_input(input)):
            _output, g = s._eval(c, _input, grad=grad, debug=debug)
            output.append(_output)
            gradinfo.append(g)
        output_ = numpy.concatenate(output, axis=1)
        if not grad:
            return output_, None
        return output_, gradinfo

    def _grad(
        self,
        coeff: ndarray,
        gradinfo: _Merge_gradinfo_t,
        dL_do: ndarray,
        *,
        debug: bool
    ) -> Tuple[ndarray, ndarray]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        dL_di: List[ndarray] = []
        dL_dc: List[ndarray] = []
        for s, c, g, _dL_do in zip(
            stages, coeffs, gradinfo, self._unpack_output(dL_do)
        ):
            _dL_di, _dL_dc = s._grad(c, g, _dL_do, debug=debug)
            dL_di.append(_dL_di)
            dL_dc.append(_dL_dc)
        return numpy.concatenate(dL_di, axis=1), numpy.concatenate(dL_dc)

    def get_constraint(self) -> Constraints:
        from scipy.linalg import block_diag  # type: ignore

        A, b, lb, ub = zip(*(s.get_constraint() for s in self.stages))
        return Constraints(
            block_diag(*A),
            numpy.concatenate(b),
            numpy.concatenate(lb),
            numpy.concatenate(ub),
        )
