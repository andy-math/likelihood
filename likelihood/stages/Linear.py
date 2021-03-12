from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from likelihood.stages.abc.Stage import Stage
from numerical.typedefs import ndarray

_Linear_gradinfo_t = ndarray


class Linear(Stage[_Linear_gradinfo_t]):
    def __init__(
        self, names: List[str], input: Sequence[int], output: Sequence[int]
    ) -> None:
        super().__init__(names, input, output)

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Linear_gradinfo_t]]:
        output = (input @ coeff).reshape((-1, 1))
        if not grad:
            return output, None
        return output, input

    def _grad(
        self, coeff: ndarray, input: _Linear_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        return dL_do * coeff, dL_do.flatten() @ input
