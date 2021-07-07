from __future__ import annotations

from typing import Callable, Tuple

import numpy
from likelihood.jit import Jitted_Function
from likelihood.stages.abc import Iterative
from likelihood.stages.abc.Stage import Constraints
from overloads.typing import ndarray


def _garch_midas_output0_generate() -> Callable[
    [ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]
]:
    def implement(coeff: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
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
            numpy.array([0.0, 0.0, 0.0, output0]),
            numpy.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, output0, output0],
                ]
            )
            / (1.0 - a - b),
            numpy.empty((0,)),
            numpy.empty((0, 3)),
        )

    return implement


def _grach_midas_eval_generate() -> Callable[
    [ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]
]:
    def implement(
        coeff: ndarray, input: ndarray, lag: ndarray, pre: ndarray
    ) -> Tuple[ndarray, ndarray]:
        c, a, b = coeff[0], coeff[1], coeff[2]
        x, err, long_turn = input[0], input[1], input[2]
        z2 = (err * err) / long_turn
        short_turn = c + a * z2 + b * lag[3]
        return numpy.array([x, 0.0, short_turn * long_turn, short_turn]), pre

    return implement


def _garch_midas_grad_generate() -> Callable[
    [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
    Tuple[ndarray, ndarray, ndarray, ndarray],
]:
    def implement(
        coeff: ndarray,
        input: ndarray,
        lag: ndarray,
        _: ndarray,
        dL_do: ndarray,
        dL_dpre: ndarray,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        c, a, b = coeff[0], coeff[1], coeff[2]
        err, long_turn = input[1], input[2]
        z2 = (err * err) / long_turn
        short_turn = c + a * z2 + b * lag[3]

        dL_dx, dL_dcompose, dL_dshort = dL_do[0], dL_do[2], dL_do[3]
        dL_dshort += dL_dcompose * long_turn
        dL_dlong = dL_dcompose * short_turn

        dL_dz2 = dL_dshort * a
        dL_derr = dL_dz2 * (2 * err / long_turn)
        dL_dlong -= dL_dz2 * (z2 / long_turn)

        return (
            dL_dshort * numpy.array([1.0, z2, lag[3]]),
            numpy.array([dL_dx, dL_derr, dL_dlong]),
            numpy.array([0.0, 0.0, 0.0, dL_dshort * b]),
            dL_dpre,
        )

    return implement


class GarchMidas(Iterative.Iterative):
    def __init__(
        _,
        names: Tuple[str, str, str],
        data_in_names: Tuple[str, str, str],
        data_out_names: Tuple[str, str, str, str],
    ) -> None:

        super().__init__(
            names,
            data_in_names,
            data_out_names,
            (),
            Jitted_Function(
                Iterative.output0_signature, (), _garch_midas_output0_generate
            ),
            Jitted_Function(Iterative.eval_signature, (), _grach_midas_eval_generate),
            Jitted_Function(Iterative.grad_signature, (), _garch_midas_grad_generate),
        )

    def get_constraints(_) -> Constraints:
        A = numpy.array([[0.0, 1.0, 1.0]])
        b = numpy.array([1.0])
        lb = numpy.array([0.0, 0.0, 0.0])
        ub = numpy.array([numpy.inf, 1.0, 1.0])
        return Constraints(A, b, lb, ub)
