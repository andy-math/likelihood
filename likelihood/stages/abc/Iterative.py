from __future__ import annotations

from abc import ABCMeta
from typing import Callable, Optional, Sequence, Tuple

import numpy
from likelihood.stages.abc.Stage import Stage
from numba import float64, njit, types  # type: ignore
from numerical.typedefs import ndarray

_Iterative_gradinfo_t = Tuple[ndarray, ndarray, ndarray, ndarray]


def _make_eval(
    output0f: Callable[[ndarray], Tuple[ndarray, ndarray]],
    evalf: Callable[[ndarray, ndarray, ndarray], ndarray],
    *,
    compile: bool
) -> Callable[
    [ndarray, ndarray, bool],
    Tuple[ndarray, Optional[_Iterative_gradinfo_t]],
]:
    if compile:
        output0f = njit(types.Tuple((float64[:], float64[:, :]))(float64[:]))(output0f)
        evalf = njit(float64[:](float64[:], float64[:], float64[:]))(evalf)

    def _eval_impl(
        coeff: ndarray, inputs: ndarray, grad: bool
    ) -> Tuple[ndarray, Optional[_Iterative_gradinfo_t]]:
        output0, d0_dc = output0f(coeff)
        nSample, nOutput = inputs.shape[0], output0.shape[0]
        outputs = numpy.ndarray((nSample, nOutput))
        outputs[0, :] = evalf(coeff, inputs[0, :], output0)
        for i in range(1, nSample):
            outputs[i, :] = evalf(coeff, inputs[i, :], outputs[i - 1, :])
        if not grad:
            return outputs, None
        return outputs, (output0, inputs, outputs, d0_dc)

    return _eval_impl


def _make_grad(
    gradf: Callable[
        [ndarray, ndarray, ndarray, ndarray, ndarray],
        Tuple[ndarray, ndarray, ndarray],
    ],
    *,
    compile: bool
) -> Callable[[ndarray, _Iterative_gradinfo_t, ndarray], Tuple[ndarray, ndarray]]:
    if compile:
        gradf = njit(
            types.UniTuple(float64[:], 3)(
                float64[:], float64[:], float64[:], float64[:], float64[:]
            )
        )(gradf)

    def _grad_impl(
        coeff: ndarray, gradinfo: _Iterative_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        output0, inputs, outputs, d0_dc = gradinfo
        nSample, nInput = inputs.shape
        dL_di = numpy.ndarray((nSample, nInput))
        dL_dc = numpy.zeros(coeff.shape)
        for i in range(nSample - 1, 0, -1):
            _dL_dc, dL_di[i, :], _dL_do = gradf(
                coeff, inputs[i, :], outputs[i - 1, :], outputs[i, :], dL_do[i, :]
            )
            dL_dc += _dL_dc
            dL_do[i - 1, :] += _dL_do

        _dL_dc, dL_di[i, :], dL_d0 = gradf(
            coeff, inputs[0, :], output0, outputs[0, :], dL_do[0, :]
        )
        dL_dc += _dL_dc + dL_d0 @ d0_dc
        return dL_di, dL_dc

    return _grad_impl


class Iterative(Stage[_Iterative_gradinfo_t], metaclass=ABCMeta):
    _eval_impl: Optional[
        Callable[
            [ndarray, ndarray, bool],
            Tuple[ndarray, Optional[_Iterative_gradinfo_t]],
        ]
    ]
    _grad_impl: Optional[
        Callable[[ndarray, _Iterative_gradinfo_t, ndarray], Tuple[ndarray, ndarray]]
    ]

    def __init__(
        self,
        names: Sequence[str],
        input: Sequence[int],
        output: Sequence[int],
        output0: Callable[[ndarray], Tuple[ndarray, ndarray]],
        eval: Callable[[ndarray, ndarray, ndarray], ndarray],
        grad: Callable[
            [ndarray, ndarray, ndarray, ndarray, ndarray],
            Tuple[ndarray, ndarray, ndarray],
        ],
        *,
        compile: bool
    ) -> None:
        super().__init__(names, input, output)
        self._eval_impl = _make_eval(output0, eval, compile=compile)
        self._grad_impl = _make_grad(grad, compile=compile)

    def _eval(
        self, coeff: ndarray, inputs: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Iterative_gradinfo_t]]:
        assert self._eval_impl is not None
        return self._eval_impl(coeff, inputs, grad)

    def _grad(
        self, coeff: ndarray, gradinfo: _Iterative_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        assert self._grad_impl is not None
        return self._grad_impl(coeff, gradinfo, dL_do)
