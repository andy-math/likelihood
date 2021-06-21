from __future__ import annotations

from typing import Callable, Tuple, Union

import numpy
from likelihood.jit import Jitted_Function
from likelihood.stages.abc import Iterative
from likelihood.stages.abc.Stage import Constraints
from numerical.typedefs import ndarray


def _iterize_output0_generate_1() -> Callable[
    [ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]
]:
    def implement(_: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return (
            numpy.empty((1,)),
            numpy.empty((1, 0)),
            numpy.empty((0,)),
            numpy.empty((0, 0)),
        )

    return implement


def _iterize_output0_generate_2() -> Callable[
    [ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]
]:
    def implement(_: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return (
            numpy.empty((2,)),
            numpy.empty((2, 0)),
            numpy.empty((0,)),
            numpy.empty((0, 0)),
        )

    return implement


def _iterize_output0_generate_3() -> Callable[
    [ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]
]:
    def implement(_: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return (
            numpy.empty((3,)),
            numpy.empty((3, 0)),
            numpy.empty((0,)),
            numpy.empty((0, 0)),
        )

    return implement


def _iterize_eval_generate() -> Callable[
    [ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]
]:
    def implement(
        coeff: ndarray, input: ndarray, lag: ndarray, pre: ndarray
    ) -> Tuple[ndarray, ndarray]:
        return input, pre

    return implement


def _iterize_grad_generate() -> Callable[
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
        return (numpy.empty((0,)), dL_do, numpy.zeros(lag.shape), dL_dpre)

    return implement


class Iterize(Iterative.Iterative):
    def __init__(
        _,
        data_in_names: Union[Tuple[str], Tuple[str, str], Tuple[str, str, str]],
        data_out_names: Union[Tuple[str], Tuple[str, str], Tuple[str, str, str]],
        input: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]],
        output: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    ) -> None:
        assert len(input) == len(output)
        assert 1 <= len(input) <= 3
        if len(input) == 1:
            _iterize_output0_generate = _iterize_output0_generate_1
        elif len(input) == 2:
            _iterize_output0_generate = _iterize_output0_generate_2
        elif len(input) == 3:
            _iterize_output0_generate = _iterize_output0_generate_3
        else:
            assert False  # pragma: no cover
        super().__init__(
            (),
            data_in_names,
            data_out_names,
            input,
            output,
            (),
            Jitted_Function(Iterative.output0_signature, (), _iterize_output0_generate),
            Jitted_Function(Iterative.eval_signature, (), _iterize_eval_generate),
            Jitted_Function(Iterative.grad_signature, (), _iterize_grad_generate),
        )

    def get_constraints(_) -> Constraints:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return Constraints(A, b, lb, ub)
