from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy
from likelihood.jit import Jitted_Function
from likelihood.stages.abc import Iterative
from numerical.typedefs import ndarray


def _tvtp_output0_generate(
    out0_f1: Callable[[ndarray], Tuple[ndarray, ndarray]],
    out0_f2: Callable[[ndarray], Tuple[ndarray, ndarray]],
) -> Callable[[ndarray], Tuple[ndarray, ndarray]]:
    def implement(coeff: ndarray) -> Tuple[ndarray, ndarray]:
        (nCoeff,) = coeff.shape
        halfCoeff = nCoeff // 2
        assert halfCoeff * 2 == nCoeff

        out0_1, dout_1 = out0_f1(coeff[:halfCoeff])
        out0_2, dout_2 = out0_f2(coeff[halfCoeff:])

        out0 = numpy.concatenate((out0_1, out0_2, numpy.array([0.5, 0.5])))

        dout = numpy.zeros(
            (
                dout_1.shape[0] + dout_2.shape[0] + 2,
                dout_1.shape[1] + dout_2.shape[1],
            )
        )

        dout[: dout_1.shape[0], : dout_1.shape[1]] = dout_1
        dout[dout_1.shape[0] : -2, dout_1.shape[1] :] = dout_2  # noqa: E203

        return out0, dout

    return implement


def _tvtp_eval_generate(
    eval_f1: Callable[[ndarray, ndarray, ndarray], ndarray],
    eval_f2: Callable[[ndarray, ndarray, ndarray], ndarray],
) -> Callable[[ndarray, ndarray, ndarray], ndarray]:
    def implement(coeff: ndarray, input: ndarray, lag: ndarray) -> ndarray:
        (nCoeff,) = coeff.shape
        halfCoeff = nCoeff // 2
        assert halfCoeff * 2 == nCoeff

        p11: float
        p22: float
        input, p11, p22 = input[:-2], input[-2], input[-1]
        (nInput,) = input.shape
        halfInput = nInput // 2
        assert halfInput * 2 == nInput

        lag_post1: float
        lag_post2: float
        lag, lag_post1, lag_post2 = lag[:-2], lag[-2], lag[-1]
        (nLag,) = lag.shape
        halfLag = nLag // 2
        assert halfLag * 2 == nLag

        rawpath11: float
        rawpath22: float
        rawpath11, rawpath12 = p11 * lag_post1, (1.0 - p22) * lag_post2
        rawpath21, rawpath22 = (1.0 - p11) * lag_post1, p22 * lag_post2

        归一化stage1: float = (rawpath11 + rawpath12) + (rawpath21 + rawpath22)

        path11 = rawpath11 / 归一化stage1
        path12 = rawpath12 / 归一化stage1
        path21 = rawpath21 / 归一化stage1
        path22 = rawpath22 / 归一化stage1

        prior1 = path11 + path12
        prior2 = path21 + path22

        contrib11 = path11 / prior1 if prior1 else 0.5
        contrib22 = path22 / prior2 if prior2 else 0.5

        lag1 = contrib11 * lag[:halfLag] + (1.0 - contrib11) * lag[halfLag:]
        lag2 = (1.0 - contrib22) * lag[:halfLag] + contrib22 * lag[halfLag:]

        out_1 = eval_f1(coeff[:halfCoeff], input[:halfInput], lag1)
        out_2 = eval_f2(coeff[halfCoeff:], input[halfInput:], lag2)

        likeli1: float = out_1[0]
        likeli2: float = out_2[0]

        rawpost1: float = prior1 * likeli1
        rawpost2: float = prior2 * likeli2

        归一化stage2 = rawpost1 + rawpost2
        if 归一化stage2 == 0:
            post1, post2, 归一化stage2 = 0.5, 0.5, 1.0
        post1, post2 = post1 / 归一化stage2, post2 / 归一化stage2

        return numpy.concatenate((out_1, out_2, numpy.array([post1, post2])))

    return implement


def _tvtp_grad_generate(
    grad_f1: Callable[
        [ndarray, ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray]
    ],
    grad_f2: Callable[
        [ndarray, ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray]
    ],
) -> Callable[
    [ndarray, ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray]
]:
    def implement(
        coeff: ndarray, input: ndarray, lag: ndarray, output: ndarray, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray]:
        (nCoeff,) = coeff.shape
        halfCoeff = nCoeff // 2
        assert halfCoeff * 2 == nCoeff

        p11: float
        p22: float
        input, p11, p22 = input[:-2], input[-2], input[-1]
        (nInput,) = input.shape
        halfInput = nInput // 2
        assert halfInput * 2 == nInput

        lag_post1: float
        lag_post2: float
        lag, lag_post1, lag_post2 = lag[:-2], lag[-2], lag[-1]
        (nLag,) = lag.shape
        halfLag = nLag // 2
        assert halfLag * 2 == nLag

        dL_dpost1: float
        dL_dpost2: float
        output, dL_do, dL_dpost1, dL_dpost2 = (
            output[:-2],
            dL_do[:-2],
            dL_do[-2],
            dL_do[-1],
        )
        (nOutput,) = output.shape
        halfOutput = nOutput // 2
        assert halfOutput * 2 == nOutput

        likeli1: float = output[0]  # output[:halfOutput][0]
        likeli2: float = output[halfOutput]  # output[halfOutput:][0]

        rawpath11: float
        rawpath22: float
        rawpath11, rawpath12 = p11 * lag_post1, (1.0 - p22) * lag_post2
        rawpath21, rawpath22 = (1.0 - p11) * lag_post1, p22 * lag_post2

        归一化stage1: float = (rawpath11 + rawpath12) + (rawpath21 + rawpath22)

        path11 = rawpath11 / 归一化stage1
        path12 = rawpath12 / 归一化stage1
        path21 = rawpath21 / 归一化stage1
        path22 = rawpath22 / 归一化stage1

        prior1 = path11 + path12
        prior2 = path21 + path22

        contrib11 = path11 / prior1 if prior1 else 0.5
        contrib22 = path22 / prior2 if prior2 else 0.5

        rawlag1 = lag[:halfLag]
        rawlag2 = lag[halfLag:]

        lag1 = contrib11 * rawlag1 + (1.0 - contrib11) * rawlag2
        lag2 = (1.0 - contrib22) * rawlag1 + contrib22 * rawlag2

        rawpost1: float = prior1 * likeli1
        rawpost2: float = prior2 * likeli2

        归一化stage2 = rawpost1 + rawpost2
        if 归一化stage2 == 0:
            rawpost1, rawpost2, 归一化stage2 = 0.5, 0.5, 1.0
        post1, post2 = rawpost1 / 归一化stage2, rawpost2 / 归一化stage2

        dL_dstage2 = -dL_dpost1 * (post1 / 归一化stage2) - dL_dpost2 * (post2 / 归一化stage2)
        dL_drawpost1 = dL_dpost1 / 归一化stage2 + dL_dstage2
        dL_drawpost2 = dL_dpost2 / 归一化stage2 + dL_dstage2
        if rawpost1 + rawpost2 == 0:
            dL_drawpost1, dL_drawpost2 = 0.0, 0.0

        dL_dprior1 = dL_drawpost1 * likeli1
        dL_dprior2 = dL_drawpost2 * likeli2

        dL_dlikeli1 = dL_drawpost1 * prior1
        dL_dlikeli2 = dL_drawpost2 * prior2

        dL_dout1, dL_dout2 = dL_do[:halfOutput], dL_do[halfOutput:]
        dL_dout1[0] += dL_dlikeli1
        dL_dout2[0] += dL_dlikeli2

        dL_dcoeff1, dL_dinput1, dL_dlag1 = grad_f1(
            coeff[:halfCoeff], input[:halfInput], lag1, output[:halfOutput], dL_dout1
        )
        dL_dcoeff2, dL_dinput2, dL_dlag2 = grad_f2(
            coeff[halfCoeff:], input[halfInput:], lag2, output[halfOutput:], dL_dout2
        )

        dL_drawlag1 = dL_dlag1 * contrib11 + dL_dlag2 * (1.0 - contrib22)
        dL_drawlag2 = dL_dlag1 * (1.0 - contrib11) + dL_dlag2 * contrib22

        dL_dcontrib11 = float(dL_dlag1 @ (rawlag1 - rawlag2))
        dL_dcontrib22 = float(dL_dlag2 @ (rawlag2 - rawlag1))

        dL_dprior1 -= dL_dcontrib11 * (contrib11 / prior1) if prior1 else 0.0
        dL_dprior2 -= dL_dcontrib22 * (contrib22 / prior2) if prior2 else 0.0

        dL_dpath11 = (dL_dcontrib11 / prior1 if prior1 else 0.0) + dL_dprior1
        dL_dpath12 = dL_dprior1
        dL_dpath21 = dL_dprior2
        dL_dpath22 = (dL_dcontrib22 / prior2 if prior2 else 0.0) + dL_dprior2

        dL_dstage1 = (
            -dL_dpath11 * (path11 / 归一化stage1)
            - dL_dpath12 * (path12 / 归一化stage1)
            - dL_dpath21 * (path21 / 归一化stage1)
            - dL_dpath22 * (path22 / 归一化stage1)
        )

        dL_drawpath11 = dL_dpath11 / 归一化stage1 + dL_dstage1
        dL_drawpath12 = dL_dpath12 / 归一化stage1 + dL_dstage1
        dL_drawpath21 = dL_dpath21 / 归一化stage1 + dL_dstage1
        dL_drawpath22 = dL_dpath22 / 归一化stage1 + dL_dstage1

        dL_dlagpost1 = dL_drawpath11 * p11 + dL_drawpath21 * (1.0 - p11)
        dL_dlagpost2 = dL_drawpath12 * (1.0 - p22) + dL_drawpath22 * p22

        dL_dp11 = dL_drawpath11 * lag_post1 - dL_drawpath21 * lag_post1
        dL_dp22 = dL_drawpath22 * lag_post2 - dL_drawpath12 * lag_post2

        dL_dcoeff = numpy.concatenate((dL_dcoeff1, dL_dcoeff2))
        dL_dinput = numpy.concatenate(
            (dL_dinput1, dL_dinput2, numpy.array([dL_dp11, dL_dp22]))
        )
        dL_drawlag = numpy.concatenate(
            (dL_drawlag1, dL_drawlag2, numpy.array([dL_dlagpost1, dL_dlagpost2]))
        )

        return (dL_dcoeff, dL_dinput, dL_drawlag)

    return implement


class MS_TVTP(Iterative.Iterative):
    mapping: List[int]

    def __init__(
        self,
        submodel: Tuple[Iterative.Iterative, Iterative.Iterative],
        sharing: List[str],
        input: Tuple[int, int],
        output: Tuple[int, int],
    ) -> None:
        assert isinstance(submodel[0], type(submodel[1]))

        for s in sharing:
            assert s in submodel[0].names and s in submodel[1].names
        for s in submodel[0].names:
            assert s in sharing or s not in submodel[1].names
        for s in submodel[1].names:
            assert s in sharing or s not in submodel[0].names

        names: List[str] = []

        mapping: List[int] = []
        for sub in submodel:
            for s in sub.names:
                found = False
                for index, n in enumerate(names):
                    if s == n:
                        mapping.append(index)
                        found = True
                        break
                if not found:
                    names.append(s)
                    mapping.append(len(names) - 1)

        super().__init__(
            names,
            tuple(submodel[0]._input_idx) + tuple(submodel[1]._input_idx) + input,
            tuple(submodel[0]._output_idx) + tuple(submodel[1]._output_idx) + output,
            Jitted_Function(
                Iterative.output0_signature,
                tuple(x._output0_scalar for x in submodel),
                _tvtp_output0_generate,
            ),
            Jitted_Function(
                Iterative.eval_signature,
                tuple(x._eval_scalar for x in submodel),
                _tvtp_eval_generate,
            ),
            Jitted_Function(
                Iterative.grad_signature,
                tuple(x._grad_scalar for x in submodel),
                _tvtp_grad_generate,
            ),
        )
        self.mapping = mapping

    def _eval(
        self, coeff: ndarray, inputs: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[Iterative._Iterative_gradinfo_t]]:
        return super()._eval(coeff[self.mapping], inputs, grad=grad, debug=debug)

    def _grad(
        self,
        coeff: ndarray,
        gradinfo: Iterative._Iterative_gradinfo_t,
        dL_do: ndarray,
        *,
        debug: bool
    ) -> Tuple[ndarray, ndarray]:
        dL_do, _dL_dc = super()._grad(coeff[self.mapping], gradinfo, dL_do, debug=debug)
        dL_dc = numpy.zeros(coeff.shape)
        for (index, d) in zip(self.mapping, _dL_dc):
            dL_dc[index] += d
        return dL_do, dL_dc

    def get_constraint(_) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return A, b, lb, ub
