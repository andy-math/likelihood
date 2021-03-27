from __future__ import annotations

from abc import ABCMeta
from typing import Callable, Optional, Sequence, Tuple

import numba  # type: ignore
import numpy
from likelihood.jit import Jitted_Function
from likelihood.stages.abc.Stage import Stage
from numba import float64, optional, types
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

_eval_generator_signature = types.Tuple(
    (float64[:, :], optional(_Iterative_gradinfo_numba))
)(float64[:], float64[:, :], numba.boolean)
_grad_generator_signature = types.Tuple((float64[:, :], float64[:]))(
    float64[:], _Iterative_gradinfo_numba, float64[:, :]
)


def _eval_generator(
    output0_func: Callable[[ndarray], Tuple[ndarray, ndarray]],
    eval_func: Callable[[ndarray, ndarray, ndarray], ndarray],
) -> Callable[
    [ndarray, ndarray, bool],
    Tuple[ndarray, Optional[_Iterative_gradinfo_t]],
]:
    def implement(
        coeff: ndarray, inputs: ndarray, grad: bool
    ) -> Tuple[ndarray, Optional[_Iterative_gradinfo_t]]:
        output0, d0_dc = output0_func(coeff)
        nSample, nOutput = inputs.shape[0], output0.shape[0]
        outputs = numpy.empty((nSample, nOutput))
        outputs[0, :] = eval_func(coeff, inputs[0, :], output0)
        for i in range(1, nSample):
            outputs[i, :] = eval_func(coeff, inputs[i, :], outputs[i - 1, :])
        if not grad:
            return outputs, None
        return outputs, (output0, inputs, outputs, d0_dc)

    return implement


def _grad_generator(
    grad_func: Callable[
        [ndarray, ndarray, ndarray, ndarray, ndarray],
        Tuple[ndarray, ndarray, ndarray],
    ],
) -> Callable[[ndarray, _Iterative_gradinfo_t, ndarray], Tuple[ndarray, ndarray]]:
    def implement(
        coeff: ndarray, gradinfo: _Iterative_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        output0, inputs, outputs, d0_dc = gradinfo
        nSample, nInput = inputs.shape
        dL_di = numpy.empty((nSample, nInput))
        dL_dc = numpy.zeros(coeff.shape)
        for i in range(nSample - 1, 0, -1):
            _dL_dc, dL_di[i, :], _dL_do = grad_func(
                coeff, inputs[i, :], outputs[i - 1, :], outputs[i, :], dL_do[i, :]
            )
            dL_dc += _dL_dc
            dL_do[i - 1, :] += _dL_do

        _dL_dc, dL_di[0, :], dL_d0 = grad_func(
            coeff, inputs[0, :], output0, outputs[0, :], dL_do[0, :]
        )
        dL_dc += _dL_dc + dL_d0 @ d0_dc
        return dL_di, dL_dc

    return implement


class Iterative(Stage[_Iterative_gradinfo_t], metaclass=ABCMeta):
    _eval_impl: Jitted_Function[
        Callable[
            [ndarray, ndarray, bool],
            Tuple[ndarray, Optional[_Iterative_gradinfo_t]],
        ]
    ]
    _grad_impl: Jitted_Function[
        Callable[[ndarray, _Iterative_gradinfo_t, ndarray], Tuple[ndarray, ndarray]]
    ]

    def __init__(
        self,
        names: Sequence[str],
        input: Sequence[int],
        output: Sequence[int],
        output0: Jitted_Function[Callable[[ndarray], Tuple[ndarray, ndarray]]],
        eval: Jitted_Function[Callable[[ndarray, ndarray, ndarray], ndarray]],
        grad: Jitted_Function[
            Callable[
                [ndarray, ndarray, ndarray, ndarray, ndarray],
                Tuple[ndarray, ndarray, ndarray],
            ]
        ],
    ) -> None:
        super().__init__(names, input, output)
        self._eval_impl = Jitted_Function(
            _eval_generator_signature, (output0, eval), _eval_generator
        )
        self._grad_impl = Jitted_Function(
            _grad_generator_signature, (grad,), _grad_generator
        )

    def _eval(
        self, coeff: ndarray, inputs: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Iterative_gradinfo_t]]:
        if debug:
            return self._eval_impl.py_func()(coeff, inputs, grad)
        return self._eval_impl.func()(coeff, inputs, grad)

    def _grad(
        self,
        coeff: ndarray,
        gradinfo: _Iterative_gradinfo_t,
        dL_do: ndarray,
        *,
        debug: bool
    ) -> Tuple[ndarray, ndarray]:
        if debug:
            return self._grad_impl.py_func()(coeff, gradinfo, dL_do)
        return self._grad_impl.func()(coeff, gradinfo, dL_do)
