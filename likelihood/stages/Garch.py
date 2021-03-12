from __future__ import annotations

from typing import Sequence, Tuple

import numpy
from likelihood.stages.abc.Iterative import Iterative
from numerical.typedefs import ndarray


class Garch(Iterative):
    def __init__(
        self, names: Sequence[str], input: Sequence[int], output: Sequence[int]
    ) -> None:
        def output0(coeff: ndarray) -> Tuple[ndarray, ndarray]:
            """
            var = c + a*var + b*var
            c = (1-a-b)var
            var = c/(1-a-b)
            """
            c, a, b = coeff
            output0 = c / (1 - a - b)
            """
            d0_dc = 1/(1-a-b)
            d0_da = c * -{ 1/(1-a-b)^2 } * -1
                  = c/[ (1-a-b)*(1-a-b) ]
            d0_db = c/[ (1-a-b)*(1-a-b) ]
            """
            return (
                numpy.array([output0]),
                numpy.array([[1.0, output0, output0]]) / (1 - a - b),
            )

        def eval(coeff: ndarray, input: ndarray, lag: ndarray) -> ndarray:
            c, a, b = coeff
            return c + a * input * input + b * lag

        def grad(
            coeff: ndarray,
            input: ndarray,
            lag: ndarray,
            _: ndarray,
            dL_do: ndarray,
        ) -> Tuple[ndarray, ndarray, ndarray]:
            """
            out = c + a*in*in + b*lag
            dcoeff = [1, in*in, lag]
            dinput = 2*a*in
            dlag   = b
            """
            c, a, b = coeff
            _input = float(input)
            return (
                dL_do * numpy.array([1.0, _input, float(lag)]),
                dL_do * 2.0 * a * input,
                dL_do * b,
            )

        super().__init__(names, input, output, output0, eval, grad)
