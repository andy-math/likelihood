from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from overloads.typing import ndarray

_Merge_gradinfo_t = List[Any]


class Merge(Stage[_Merge_gradinfo_t]):
    def __init__(self, submodels: Tuple[Stage[Any], ...]) -> None:
        super().__init__((), (), (), submodels)

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Merge_gradinfo_t]]:
        output: List[ndarray] = []
        gradinfo: List[Optional[Any]] = []
        for s in self.submodels:
            _output, g = s._eval(
                coeff[s.coeff_index], input[:, s.data_in_index], grad=grad, debug=debug
            )
            output.append(_output)
            gradinfo.append(g)
        output_: ndarray = numpy.concatenate(output, axis=1)  # type: ignore
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
        dL_di: List[ndarray] = []
        dL_dc: List[ndarray] = []
        for s, g in zip(self.submodels, gradinfo):
            _dL_di, _dL_dc = s._grad(
                coeff[s.coeff_index], g, dL_do[:, s.data_out_index], debug=debug
            )
            dL_di.append(_dL_di)
            dL_dc.append(_dL_dc)
        return (
            numpy.concatenate(dL_di, axis=1),  # type: ignore
            numpy.concatenate(dL_dc),  # type: ignore
        )

    def get_constraints(self) -> Constraints:
        return Constraints(
            numpy.empty((0, len(self.coeff_names))),
            numpy.empty((0,)),
            numpy.full((len(self.coeff_names),), -numpy.inf),
            numpy.full((len(self.coeff_names),), numpy.inf),
        )
