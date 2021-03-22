from __future__ import annotations

from abc import ABCMeta
from typing import Callable, Optional, Sequence, Tuple

import numba  # type: ignore
import numpy
from likelihood.stages.abc.Stage import Stage
from numba import float64, njit, optional, types
from numerical.typedefs import ndarray

_Iterative_gradinfo_t = Tuple[ndarray, ndarray, ndarray, ndarray]
_Iterative_gradinfo_numba = types.Tuple(
    (float64[:], float64[:, :], float64[:, :], float64[:, ::1])
)
output0_signature = types.Tuple((float64[:], float64[:, ::1]))(float64[:])
eval_signature = float64[:](float64[:], float64[:], float64[:])
grad_signature = types.UniTuple(float64[::1], 3)(
    float64[:], float64[:], float64[:], float64[:], float64[:]
)


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
        output0f = njit(output0_signature)(output0f)
        evalf = njit(eval_signature)(evalf)

    def _eval_impl(
        coeff: ndarray, inputs: ndarray, grad: bool
    ) -> Tuple[ndarray, Optional[_Iterative_gradinfo_t]]:
        output0, d0_dc = output0f(coeff)
        nSample, nOutput = inputs.shape[0], output0.shape[0]
        outputs = numpy.empty((nSample, nOutput))
        outputs[0, :] = evalf(coeff, inputs[0, :], output0)
        for i in range(1, nSample):
            outputs[i, :] = evalf(coeff, inputs[i, :], outputs[i - 1, :])
        if not grad:
            return outputs, None
        return outputs, (output0, inputs, outputs, d0_dc)

    if compile:
        _eval_impl = njit(
            types.Tuple((float64[:, :], optional(_Iterative_gradinfo_numba)))(
                float64[:], float64[:, :], numba.boolean
            )
        )(_eval_impl)

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
        gradf = njit(grad_signature)(gradf)

    def _grad_impl(
        coeff: ndarray, gradinfo: _Iterative_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        output0, inputs, outputs, d0_dc = gradinfo
        nSample, nInput = inputs.shape
        dL_di = numpy.empty((nSample, nInput))
        dL_dc = numpy.zeros(coeff.shape)
        for i in range(nSample - 1, 0, -1):
            _dL_dc, dL_di[i, :], _dL_do = gradf(
                coeff, inputs[i, :], outputs[i - 1, :], outputs[i, :], dL_do[i, :]
            )
            dL_dc += _dL_dc
            dL_do[i - 1, :] += _dL_do

        _dL_dc, dL_di[0, :], dL_d0 = gradf(
            coeff, inputs[0, :], output0, outputs[0, :], dL_do[0, :]
        )
        dL_dc += _dL_dc + dL_d0 @ d0_dc
        return dL_di, dL_dc

    if compile:
        _grad_impl = njit(
            types.Tuple((float64[:, :], float64[:]))(
                float64[:], _Iterative_gradinfo_numba, float64[:, :]
            )
        )(_grad_impl)

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
