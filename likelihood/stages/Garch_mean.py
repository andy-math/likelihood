from __future__ import annotations

from typing import Tuple

import numpy
from likelihood.stages.abc.Iterative import Iterative
from numerical.typedefs import ndarray


def _garch_mean_output0_impl(coeff: ndarray) -> Tuple[ndarray, ndarray]:
    """
    var = c + a*var + b*var
    c = (1-a-b)var
    var = c/(1-a-b)
    """
    c, a, b = coeff[0], coeff[1], coeff[2]
    output0 = c / (1.0 - a - b)
    """
    d0_dc = 1/(1-a-b)
    d0_da = c * -{ 1/(1-a-b)^2 } * -1
          = c/[ (1-a-b)*(1-a-b) ]
    d0_db = c/[ (1-a-b)*(1-a-b) ]
    """
    return (
        numpy.array([0.0, 0.0, output0]),
        numpy.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, output0, output0],
            ]
        )
        / (1.0 - a - b),
    )


def _grach_mean_eval_impl(coeff: ndarray, input: ndarray, lag: ndarray) -> ndarray:
    c, a, b = coeff[0], coeff[1], coeff[2]
    x, mu = input[0], input[1]
    EX, EX2 = lag[1], lag[2]

    hlag = max(EX2 - EX * EX, 0.0)
    err = x - mu
    h = c + a * err * err + b * hlag
    EX, EX2 = mu, mu * mu + h  # type: ignore
    return numpy.array([hlag, EX, EX2])


def _garch_mean_grad_impl(
    coeff: ndarray, input: ndarray, lag: ndarray, _: ndarray, dL_do: ndarray
) -> Tuple[ndarray, ndarray, ndarray]:
    dL_dhlag: float
    dL_dmu: float
    dL_dEX2: float

    x, mu = input[0], input[1]
    dL_dhlag, dL_dmu, dL_dEX2 = dL_do[0], dL_do[1], dL_do[2]

    dL_dh = dL_dEX2
    dL_dmu += dL_dEX2 * (2.0 * mu)

    a, b = coeff[1], coeff[2]
    err = x - mu
    dL_derr = dL_dh * a * 2.0 * err
    dL_dinput = numpy.array([dL_derr, dL_dmu - dL_derr])

    EX, EX2 = lag[1], lag[2]
    h = max(EX2 - EX * EX, 0.0)
    dL_dcoeff = dL_dh * numpy.array([1.0, err * err, h])

    dL_dhlag += dL_dh * b
    dL_dEX2 = dL_dhlag
    dL_dEX = -dL_dhlag * 2 * EX
    dL_dlag = numpy.array([0.0, dL_dEX, dL_dEX2])
    return (dL_dcoeff, dL_dinput, dL_dlag)


class Garch_mean(Iterative):
    def __init__(
        self,
        names: Tuple[str, str, str],
        input: Tuple[int, int],
        output: Tuple[int, int, int],
        *,
        compile: bool
    ) -> None:

        super().__init__(
            names,
            input,
            output,
            _garch_mean_output0_impl,
            _grach_mean_eval_impl,
            _garch_mean_grad_impl,
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