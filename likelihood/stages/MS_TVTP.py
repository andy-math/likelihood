from __future__ import annotations

import math
import sys
from typing import Callable, Dict, Literal, Tuple

import numpy
from numba import float64  # type: ignore

from likelihood.jit import Jitted_Function
from likelihood.stages.abc import Iterative, Logpdf
from likelihood.stages.abc.Stage import Constraints
from overloads.typedefs import ndarray

_eps = sys.float_info.epsilon
_realmax = sys.float_info.max


def _tvtp_output0_generate(
    out0_f1: Callable[[ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]],
    out0_f2: Callable[[ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]],
) -> Callable[[ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]]:
    def implement(coeff: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        (nCoeff,) = coeff.shape
        halfCoeff = nCoeff // 2
        assert halfCoeff * 2 == nCoeff

        out0_1, dout_1, pre_1, dpre_1 = out0_f1(coeff[:halfCoeff])
        out0_2, dout_2, pre_2, dpre_2 = out0_f2(coeff[halfCoeff:])

        out0: ndarray = numpy.concatenate(
            (numpy.array([0.0, 0.0, 0.0, 0.5, 0.5]), out0_1, out0_2)
        )
        pre: ndarray = numpy.concatenate((pre_1, pre_2))

        (nOut,) = out0_1.shape
        (nPre,) = pre_1.shape

        dout = numpy.zeros((nOut * 2 + 5, halfCoeff * 2))
        dout[5 : (nOut + 5), :halfCoeff] = dout_1
        dout[(nOut + 5) : (2 * nOut + 5), halfCoeff:] = dout_2

        dpre = numpy.zeros((nPre * 2, halfCoeff * 2))
        dpre[:nPre, :halfCoeff] = dpre_1
        dpre[nPre:, halfCoeff:] = dpre_2

        return out0, dout, pre, dpre

    return implement


def _tvtp_eval_generate(
    eval_f1: Callable[[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    eval_f2: Callable[[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    likeli_provider: Callable[[ndarray], float],
) -> Callable[[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]:
    def implement(
        coeff: ndarray, input: ndarray, lag: ndarray, pre: ndarray
    ) -> Tuple[ndarray, ndarray]:
        (nCoeff,) = coeff.shape
        halfCoeff = nCoeff // 2
        assert halfCoeff * 2 == nCoeff

        p11: float
        p22: float
        p11, p22, input = input[0], input[1], input[2:]
        (nInput,) = input.shape
        halfInput = nInput // 2
        assert halfInput * 2 == nInput

        lag_post1: float
        lag_post2: float
        lag_post1, lag_post2, lag = lag[3], lag[4], lag[5:]
        (nLag,) = lag.shape
        halfLag = nLag // 2
        assert halfLag * 2 == nLag

        (nPre,) = pre.shape
        halfPre = nPre // 2
        assert halfPre * 2 == nPre

        rawpath11: float
        rawpath22: float
        rawpath11, rawpath12 = p11 * lag_post1, (1.0 - p22) * lag_post2
        rawpath21, rawpath22 = (1.0 - p11) * lag_post1, p22 * lag_post2

        归一化stage1: float = (rawpath11 + rawpath12) + (rawpath21 + rawpath22)
        # 如果归一化stage1出现0，就是前序likelihood跑飞了
        if 归一化stage1 == 0:
            归一化stage1 = 1

        path11 = rawpath11 / 归一化stage1
        path12 = rawpath12 / 归一化stage1
        path21 = rawpath21 / 归一化stage1
        path22 = rawpath22 / 归一化stage1

        prior1 = path11 + path12
        prior2 = path21 + path22

        contrib11 = path11 / prior1 if prior1 > _eps else 0.5
        contrib22 = path22 / prior2 if prior2 > _eps else 0.5

        rawlag1 = lag[:halfLag]
        rawlag2 = lag[halfLag:]

        lag1 = contrib11 * rawlag1 + (1.0 - contrib11) * rawlag2
        lag2 = (1.0 - contrib22) * rawlag1 + contrib22 * rawlag2

        out_1, pre[:halfPre] = eval_f1(
            coeff[:halfCoeff], input[:halfInput], lag1, pre[:halfPre]
        )
        out_2, pre[halfPre:] = eval_f2(
            coeff[halfCoeff:], input[halfInput:], lag2, pre[halfPre:]
        )

        loglikeli1: float = likeli_provider(out_1)
        loglikeli2: float = likeli_provider(out_2)

        # 寻找一个尽可能小的倍数times=exp(loglikeli_offset)
        # 使得times*pdf1 == 1或者times*pdf2 == 1
        # 避免pdf太小出现0
        if loglikeli1 >= loglikeli2:
            loglikeli_offset = -loglikeli1
        else:
            loglikeli_offset = -loglikeli2

        likeli1 = math.exp(loglikeli1 + loglikeli_offset)
        likeli2 = math.exp(loglikeli2 + loglikeli_offset)

        # rawpost1, rawpost2, 归一化stage2都是乘以倍数之后的
        rawpost1: float = prior1 * likeli1
        rawpost2: float = prior2 * likeli2
        归一化stage2 = rawpost1 + rawpost2

        if prior1 == 0 or prior2 == 0:
            # 一侧先验概率为0，则Bayes filter输入输出结果不变。
            # 此时后验概率直接等于先验概率
            post1, post2 = prior1, prior2
        else:
            # 分子分母同时扩大了相同倍数，会被消去
            post1, post2 = rawpost1 / 归一化stage2, rawpost2 / 归一化stage2
        # 从对数之后的似然函数里消去倍数
        likelihood = math.log(归一化stage2) - loglikeli_offset

        EX2_1 = out_1[2] + out_1[1] * out_1[1]
        EX2_2 = out_2[2] + out_2[1] * out_2[1]
        EX = prior1 * out_1[1] + prior2 * out_2[1]
        var = max((prior1 * EX2_1 + prior2 * EX2_2) - EX * EX, 0.0)

        return (
            numpy.concatenate(
                (
                    numpy.array(
                        [
                            likelihood,
                            EX,
                            var,
                            post1,
                            post2,
                        ]
                    ),
                    out_1,
                    out_2,
                )
            ),
            pre,
        )

    return implement


def _tvtp_grad_generate(
    grad_f1: Callable[
        [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
        Tuple[ndarray, ndarray, ndarray, ndarray],
    ],
    grad_f2: Callable[
        [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
        Tuple[ndarray, ndarray, ndarray, ndarray],
    ],
    likeli_provider: Callable[[ndarray], float],
    likeli_gradient: Callable[[ndarray, float, float], ndarray],
) -> Callable[
    [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
    Tuple[ndarray, ndarray, ndarray, ndarray],
]:
    def implement(
        coeff: ndarray,
        input: ndarray,
        lag: ndarray,
        output: ndarray,
        dL_do: ndarray,
        dL_dpre: ndarray,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        (nCoeff,) = coeff.shape
        halfCoeff = nCoeff // 2
        assert halfCoeff * 2 == nCoeff

        p11: float
        p22: float
        p11, p22, input = input[0], input[1], input[2:]
        (nInput,) = input.shape
        halfInput = nInput // 2
        assert halfInput * 2 == nInput

        lag_post1: float
        lag_post2: float
        lag_post1, lag_post2, lag = lag[3], lag[4], lag[5:]
        (nLag,) = lag.shape
        halfLag = nLag // 2
        assert halfLag * 2 == nLag

        dL_dpost1: float
        dL_dpost2: float
        output, dL_dlike, dL_dEX, dL_dvar, dL_dpost1, dL_dpost2, dL_do = (
            output[5:],
            dL_do[0],
            dL_do[1],
            dL_do[2],
            dL_do[3],
            dL_do[4],
            dL_do[5:],
        )
        (nOutput,) = output.shape
        halfOutput = nOutput // 2
        assert halfOutput * 2 == nOutput

        (nPre,) = dL_dpre.shape
        halfPre = nPre // 2
        assert halfPre * 2 == nPre

        out1 = output[:halfOutput]
        out2 = output[halfOutput:]

        dL_dout1 = dL_do[:halfOutput]
        dL_dout2 = dL_do[halfOutput:]

        loglikeli1: float = likeli_provider(out1)
        loglikeli2: float = likeli_provider(out2)

        rawpath11: float
        rawpath22: float
        rawpath11, rawpath12 = p11 * lag_post1, (1.0 - p22) * lag_post2
        rawpath21, rawpath22 = (1.0 - p11) * lag_post1, p22 * lag_post2

        归一化stage1: float = (rawpath11 + rawpath12) + (rawpath21 + rawpath22)
        # 如果归一化stage1出现0，就是前序likelihood跑飞了
        stage1_patched = False
        if 归一化stage1 == 0:
            归一化stage1 = 1
            stage1_patched = True

        path11 = rawpath11 / 归一化stage1
        path12 = rawpath12 / 归一化stage1
        path21 = rawpath21 / 归一化stage1
        path22 = rawpath22 / 归一化stage1

        prior1 = path11 + path12
        prior2 = path21 + path22

        if prior1 > _eps:
            contrib11 = path11 / prior1
            prior1_patched = False
        else:
            contrib11 = 0.5
            prior1_patched = True

        if prior2 > _eps:
            contrib22 = path22 / prior2
            prior2_patched = False
        else:
            contrib22 = 0.5
            prior2_patched = True

        rawlag1 = lag[:halfLag]
        rawlag2 = lag[halfLag:]

        lag1 = contrib11 * rawlag1 + (1.0 - contrib11) * rawlag2
        lag2 = (1.0 - contrib22) * rawlag1 + contrib22 * rawlag2

        # 寻找一个尽可能小的倍数times=exp(loglikeli_offset)
        # 使得times*pdf1 == 1或者times*pdf2 == 1
        # 避免pdf太小出现0
        if loglikeli1 >= loglikeli2:
            loglikeli_offset = -loglikeli1
        else:
            loglikeli_offset = -loglikeli2

        likeli1 = math.exp(loglikeli1 + loglikeli_offset)
        likeli2 = math.exp(loglikeli2 + loglikeli_offset)

        # rawpost1, rawpost2, 归一化stage2都是乘以倍数之后的
        rawpost1: float = prior1 * likeli1
        rawpost2: float = prior2 * likeli2
        归一化stage2 = rawpost1 + rawpost2

        if prior1 == 0 or prior2 == 0:
            post_shortpath = True
            # 一侧先验概率为0，则Bayes filter输入输出结果不变。
            # 此时后验概率直接等于先验概率
            post1, post2 = prior1, prior2
        else:
            post_shortpath = False
            # 分子分母同时扩大了相同倍数，会被消去
            post1, post2 = rawpost1 / 归一化stage2, rawpost2 / 归一化stage2
        # 从对数之后的似然函数里消去倍数
        # likelihood = math.log(归一化stage2) - loglikeli_offset

        EX2_1 = out1[2] + out1[1] * out1[1]
        EX2_2 = out2[2] + out2[1] * out2[1]
        EX = prior1 * out1[1] + prior2 * out2[1]
        # var = max((prior1 * EX2_1 + prior2 * EX2_2) - EX * EX, 0.0)

        dL_dEX2_1 = dL_dvar * prior1
        dL_dEX2_2 = dL_dvar * prior2
        dL_dEX += -dL_dvar * (2 * EX)

        dL_dprior1, dL_dprior2 = 0.0, 0.0
        dL_dstage2 = dL_dlike / 归一化stage2
        dL_dlikeoffset = -dL_dlike
        if post_shortpath:
            dL_dprior1 += dL_dpost1
            dL_dprior2 += dL_dpost2
            dL_drawpost1, dL_drawpost2 = dL_dstage2, dL_dstage2
        else:
            dL_dstage2 -= dL_dpost1 * (post1 / 归一化stage2)
            dL_dstage2 -= dL_dpost2 * (post2 / 归一化stage2)
            dL_drawpost1 = dL_dpost1 / 归一化stage2 + dL_dstage2
            dL_drawpost2 = dL_dpost2 / 归一化stage2 + dL_dstage2

        dL_dprior1 += dL_drawpost1 * likeli1 + dL_dvar * EX2_1 + dL_dEX * out1[1]
        dL_dprior2 += dL_drawpost2 * likeli2 + dL_dvar * EX2_2 + dL_dEX * out2[1]

        dL_dlikeli1 = dL_drawpost1 * prior1
        dL_dlikeli2 = dL_drawpost2 * prior2

        dL_dloglikeli1 = dL_dlikeli1 * likeli1
        dL_dlikeoffset += dL_dloglikeli1
        dL_dloglikeli2 = dL_dlikeli2 * likeli2
        dL_dlikeoffset += dL_dloglikeli2

        if loglikeli1 >= loglikeli2:
            dL_dloglikeli1 -= dL_dlikeoffset
        else:
            dL_dloglikeli2 -= dL_dlikeoffset

        dL_dout1 += likeli_gradient(out1, likeli1, dL_dloglikeli1)
        dL_dout2 += likeli_gradient(out2, likeli2, dL_dloglikeli2)
        dL_dout1[1] += dL_dEX * prior1 + dL_dEX2_1 * (2 * out1[1])
        dL_dout2[1] += dL_dEX * prior2 + dL_dEX2_2 * (2 * out2[1])
        dL_dout1[2] += dL_dEX2_1
        dL_dout2[2] += dL_dEX2_2

        dL_dcoeff1, dL_dinput1, dL_dlag1, dL_dpre[:halfPre] = grad_f1(
            coeff[:halfCoeff],
            input[:halfInput],
            lag1,
            out1,
            dL_dout1,
            dL_dpre[:halfPre],
        )
        dL_dcoeff2, dL_dinput2, dL_dlag2, dL_dpre[halfPre:] = grad_f2(
            coeff[halfCoeff:],
            input[halfInput:],
            lag2,
            out2,
            dL_dout2,
            dL_dpre[halfPre:],
        )

        dL_drawlag1: ndarray = dL_dlag2 * (1.0 - contrib22) + dL_dlag1 * contrib11
        dL_drawlag2: ndarray = dL_dlag1 * (1.0 - contrib11) + dL_dlag2 * contrib22

        dL_dcontrib11 = float(dL_dlag1 @ (rawlag1 - rawlag2))
        dL_dcontrib22 = float(dL_dlag2 @ (rawlag2 - rawlag1))

        dL_dprior1 -= 0.0 if prior1_patched else dL_dcontrib11 * (contrib11 / prior1)
        dL_dprior2 -= 0.0 if prior2_patched else dL_dcontrib22 * (contrib22 / prior2)

        dL_dpath11 = (0.0 if prior1_patched else dL_dcontrib11 / prior1) + dL_dprior1
        dL_dpath12 = dL_dprior1
        dL_dpath21 = dL_dprior2
        dL_dpath22 = (0.0 if prior2_patched else dL_dcontrib22 / prior2) + dL_dprior2

        if stage1_patched:
            dL_drawpath11 = 0.0
            dL_drawpath12 = 0.0
            dL_drawpath21 = 0.0
            dL_drawpath22 = 0.0
        else:
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

        dL_dcoeff: ndarray = numpy.concatenate((dL_dcoeff1, dL_dcoeff2))
        dL_dinput: ndarray = numpy.concatenate(
            (numpy.array([dL_dp11, dL_dp22]), dL_dinput1, dL_dinput2)
        )
        dL_drawlag: ndarray = numpy.concatenate(
            (
                numpy.array([0.0, 0.0, 0.0, dL_dlagpost1, dL_dlagpost2]),
                dL_drawlag1,
                dL_drawlag2,
            )
        )

        return (dL_dcoeff, dL_dinput, dL_drawlag, dL_dpre)

    return implement


def normpdf_provider() -> Callable[[ndarray], float]:
    def implement(output: ndarray) -> float:
        """
        pdf = 1/sqrt(2*pi*var)*exp(-(x-mu)^2/(2*var))
        log pdf = -1/2(log(2) + log(pi) + log(var) + (x-mu)^2/var)
        """
        x, mu, var = output[0], output[1], output[2]
        err = x - mu
        normpdf = -(1.0 / 2.0) * (
            math.log(2) + math.log(math.pi) + math.log(var) + (err * err) / var
        )
        return normpdf  # type: ignore

    return implement


def normpdf_provider_gradient() -> Callable[[ndarray, float, float], ndarray]:
    def implement(output: ndarray, normpdf: float, dL_dpdf: float) -> ndarray:
        """
        d{log pdf}/derr = -err/var
        d{log pdf}/dvar = -(1/2)(1/var - (err*err)/(var*var))
                        = (1/2)((err/var)*(err/var) - 1/var)
        """
        x, mu, var = output[0], output[1], output[2]
        err = x - mu
        z = err / var
        dL_doutput = numpy.zeros(output.shape)

        dL_derr = dL_dpdf * -z
        dL_dvar = dL_dpdf * ((1.0 / 2.0) * (z * z - 1.0 / var))

        dL_doutput[0] = dL_derr
        dL_doutput[1] = -dL_derr
        dL_doutput[2] = dL_dvar
        return dL_doutput

    return implement


_provider_signature = float64(float64[:])
_provider_gradient_signature = float64[:](float64[:], float64, float64)


providers: Dict[
    Literal["normpdf"],
    Tuple[
        Jitted_Function[Callable[[ndarray], float]],
        Jitted_Function[Callable[[ndarray, float, float], ndarray]],
    ],
] = {
    "normpdf": (
        Jitted_Function(_provider_signature, (), normpdf_provider),
        Jitted_Function(_provider_gradient_signature, (), normpdf_provider_gradient),
    ),
}


class MS_TVTP(Iterative.Iterative, Logpdf.Logpdf[Iterative._Signature.GradInfo]):
    def __init__(
        self,
        submodels: Tuple[Iterative.Iterative, Iterative.Iterative],
        provider: Tuple[
            Jitted_Function[Callable[[ndarray], float]],
            Jitted_Function[Callable[[ndarray, float, float], ndarray]],
        ],
        data_in_names: Tuple[str, str],
        data_out_names: Tuple[str, str, str, str, str],
    ) -> None:
        assert isinstance(submodels[0], type(submodels[1]))
        assert isinstance(submodels[1], type(submodels[0]))
        assert len(submodels[0].data_in_names) == len(submodels[1].data_in_names)
        assert len(submodels[0].data_out_names) == len(submodels[1].data_out_names)

        super().__init__(
            (),
            data_in_names,
            data_out_names,
            submodels,
            Jitted_Function(
                Iterative._Numba.Output0,
                tuple(x._output0_scalar for x in submodels),
                _tvtp_output0_generate,
            ),
            Jitted_Function(
                Iterative._Numba.Eval,
                tuple(x._eval_scalar for x in submodels) + provider[:1],
                _tvtp_eval_generate,
            ),
            Jitted_Function(
                Iterative._Numba.Grad,
                tuple(x._grad_scalar for x in submodels) + provider,
                _tvtp_grad_generate,
            ),
        )

    def get_constraints(self) -> Constraints:
        return Constraints(
            numpy.empty((0, len(self.coeff_names))),
            numpy.empty((0,)),
            numpy.full((len(self.coeff_names),), -numpy.inf),
            numpy.full((len(self.coeff_names),), numpy.inf),
        )
