from __future__ import annotations

from abc import ABCMeta
from typing import Callable, Optional, Tuple

import numba  # type: ignore
import numpy
from likelihood.jit import JittedFunction, _signature_t
from likelihood.stages.abc.Stage import Stage
from numba import float64, optional, types
from overloads.typedefs import ndarray


class _Signature:
    GradInfo = Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
    Output0 = Callable[[ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]]
    Eval = Callable[[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]
    LoopEvalRet = Tuple[ndarray, Optional[GradInfo]]
    LoopEval = Callable[[ndarray, ndarray, bool], LoopEvalRet]
    Grad = Callable[
        [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
        Tuple[ndarray, ndarray, ndarray, ndarray],
    ]
    LoopGrad = Callable[[ndarray, GradInfo, ndarray], Tuple[ndarray, ndarray]]


class _Numba:
    GradInfo = _signature_t(
        types.Tuple(
            (
                float64[::1],
                float64[:, ::1],
                float64[:, ::1],
                float64[:, ::1],
                float64[:, ::1],
            )
        )
    )
    Output0 = _signature_t(
        types.Tuple((float64[::1], float64[:, ::1], float64[::1], float64[:, ::1]))(
            float64[::1]
        )
    )
    Eval = _signature_t(
        types.UniTuple(float64[::1], 2)(
            float64[::1], float64[::1], float64[::1], float64[::1]
        )
    )
    Grad = _signature_t(
        types.UniTuple(float64[::1], 4)(
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
        )
    )
    LoopEval = _signature_t(
        types.Tuple((float64[:, ::1], optional(GradInfo)))(
            float64[::1], float64[:, ::1], numba.boolean
        )
    )
    LoopGrad = _signature_t(
        types.Tuple((float64[:, ::1], float64[::1]))(
            float64[::1], GradInfo, float64[:, ::1]
        )
    )


def _eval_generator(
    output0_func: _Signature.Output0, eval_func: _Signature.Eval
) -> _Signature.LoopEval:
    def implement(
        coeff: ndarray, inputs: ndarray, grad: bool
    ) -> _Signature.LoopEvalRet:
        output0, d0_dc, preserve, dpre_dc = output0_func(coeff)
        nSample, nOutput = inputs.shape[0], output0.shape[0]
        outputs = numpy.empty((nSample, nOutput))
        outputs[0, :], preserve = eval_func(coeff, inputs[0, :], output0, preserve)
        for i in range(1, nSample):
            outputs[i, :], preserve = eval_func(
                coeff, inputs[i, :], outputs[i - 1, :], preserve
            )
        if not grad:
            return outputs, None
        return outputs, (output0, inputs, outputs, d0_dc, dpre_dc)

    return implement


def _grad_generator(grad_func: _Signature.Grad) -> _Signature.LoopGrad:
    def implement(
        coeff: ndarray, gradinfo: _Signature.GradInfo, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        output0, inputs, outputs, d0_dc, dpre_dc = gradinfo
        nSample, nInput = inputs.shape
        dL_di = numpy.empty((nSample, nInput))
        dL_dc = numpy.zeros(coeff.shape)
        dL_dpre = numpy.zeros((dpre_dc.shape[0],))
        for i in range(nSample - 1, 0, -1):
            _dL_dc, dL_di[i, :], _dL_do, dL_dpre = grad_func(
                coeff,
                inputs[i, :],
                outputs[i - 1, :],
                outputs[i, :],
                dL_do[i, :],
                dL_dpre,
            )
            dL_dc += _dL_dc
            dL_do[i - 1, :] += _dL_do

        _dL_dc, dL_di[0, :], dL_d0, dL_dpre = grad_func(
            coeff, inputs[0, :], output0, outputs[0, :], dL_do[0, :], dL_dpre
        )
        dL_dc += _dL_dc + dL_d0 @ d0_dc + dL_dpre @ dpre_dc  # type: ignore
        return dL_di, dL_dc

    return implement


class Iterative(Stage[_Signature.GradInfo], metaclass=ABCMeta):
    _eval_impl: JittedFunction[_Signature.LoopEval]
    _grad_impl: JittedFunction[_Signature.LoopGrad]

    _output0_scalar: JittedFunction[_Signature.Output0]
    _eval_scalar: JittedFunction[_Signature.Eval]
    _grad_scalar: JittedFunction[_Signature.Grad]

    def __init__(
        self,
        names: Tuple[str, ...],
        data_in_names: Tuple[str, ...],
        data_out_names: Tuple[str, ...],
        submodels: Tuple[Iterative, ...],
        output0: JittedFunction[_Signature.Output0],
        eval: JittedFunction[_Signature.Eval],
        grad: JittedFunction[_Signature.Grad],
    ) -> None:
        super().__init__(names, data_in_names, data_out_names, submodels)
        self._eval_impl = JittedFunction(
            _Numba.LoopEval, (output0, eval), _eval_generator
        )
        self._grad_impl = JittedFunction(_Numba.LoopGrad, (grad,), _grad_generator)
        self._output0_scalar = output0
        self._eval_scalar = eval
        self._grad_scalar = grad

    def _eval(
        self, coeff: ndarray, inputs: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Signature.GradInfo]]:
        if debug:
            return self._eval_impl.py_func()(coeff, inputs, grad)
        if numpy.isfortran(inputs):
            inputs = numpy.ascontiguousarray(inputs)
        return self._eval_impl.func()(coeff, inputs, grad)

    def _grad(
        self,
        coeff: ndarray,
        gradinfo: _Signature.GradInfo,
        dL_do: ndarray,
        *,
        debug: bool
    ) -> Tuple[ndarray, ndarray]:
        if debug:
            return self._grad_impl.py_func()(coeff, gradinfo, dL_do)
        if numpy.isfortran(dL_do):
            dL_do = numpy.ascontiguousarray(dL_do)
        return self._grad_impl.func()(coeff, gradinfo, dL_do)
