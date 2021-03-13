from __future__ import annotations

from typing import Tuple

import numpy
from likelihood.stages.abc.Iterative import Iterative
from numerical.typedefs import ndarray


def _garch_output0_impl(coeff: ndarray) -> Tuple[ndarray, ndarray]:
    """
    var = c + a*var + b*var
    c = (1-a-b)var
    var = c/(1-a-b)
    """
    c, a, b = coeff[0], coeff[1], coeff[2]
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


def _grach_eval_impl(coeff: ndarray, input: ndarray, lag: ndarray) -> ndarray:
    c, a, b = coeff[0], coeff[1], coeff[2]
    return numpy.array([c + a * input[0] * input[0] + b * lag[0]])


def _garch_grad_impl(
    coeff: ndarray, input: ndarray, lag: ndarray, _: ndarray, dL_do: ndarray
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    out = c + a*in*in + b*lag
    dcoeff = [1, in*in, lag]
    dinput = 2*a*in
    dlag   = b
    """
    a, b = coeff[1], coeff[2]
    _input = input[0]
    return (
        dL_do[0] * numpy.array([1.0, _input * _input, lag[0]]),
        dL_do * 2.0 * a * _input,
        dL_do * b,
    )


class Garch(Iterative):
    def __init__(
        self, names: Tuple[str, str, str], input: int, output: int, *, compile: bool
    ) -> None:

        super().__init__(
            names,
            (input,),
            (output,),
            _garch_output0_impl,
            _grach_eval_impl,
            _garch_grad_impl,
            compile=compile,
        )

    def get_constraint(
        self,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        A = numpy.array([[0.0, 1.0, 1.0]])
        b = numpy.array([1.0])
        lb = numpy.array([0.0, 0.0, 0.0])
        ub = numpy.array([numpy.inf, 1.0, 1.0])
        return A, b, lb, ub
