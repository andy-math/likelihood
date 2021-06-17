# -*- coding: utf-8 -*-
import math

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.GarchMidas import GarchMidas
from likelihood.stages.LogNormpdf_var import LogNormpdf_var
from likelihood.stages.Midas_exp import Midas_exp
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region

from tests.common import nll2func


def generate(coeff: ndarray, n: int, k: int, seed: int = 0) -> ndarray:
    omega, c, a, b = coeff
    kernel = omega ** numpy.arange(1.0, k + 1.0)[::-1]
    kernel = kernel / numpy.sum(kernel)

    numpy.random.seed(seed)
    short = c / (1 - a - b)
    x = numpy.zeros((n, 1))
    for i in range(k):
        x[0] = numpy.random.normal(loc=0, scale=math.sqrt(short), size=1)
    for i in range(k, n):
        long = kernel @ (x[i - k : i] * x[i - k : i])  # noqa: E203
        short = c + a * (x[i - 1] * x[i - 1] / long) + b * short
        x[i] = numpy.random.normal(loc=0, scale=math.sqrt(long * short), size=1)
    return x


def run_once(coeff: ndarray, n: int, k: int, seed: int = 0, times: int = 10) -> None:
    x = generate(coeff, n + times * k, k, seed=seed)[times * k :]  # noqa: E203
    x, y = x[:-1, :], x[1:, :]
    input = numpy.concatenate((y, x, x * x, numpy.zeros(y.shape)), axis=1)
    beta0 = numpy.array([0.8, 0.1, 0.1, 0.8])

    stage1 = Midas_exp("omega", ("long",), ("long",), (2,), (2,), k=k)
    stage2 = GarchMidas(
        ("c", "a", "b"),
        ("Y", "variance", "long"),
        ("Y", "drop", "variance", "long"),
        (0, 1, 2),
        (0, 3, 1, 2),
    )
    stage3 = LogNormpdf_var(("Y", "variance"), ("Y", "variance"), (0, 1), (0, 1))

    nll = likelihood.negLikelihood(
        ("omega", "c", "a", "b"),
        ("Y", "variance", "long", "drop"),
        (stage1, stage2, stage3),
        None,
        nvars=4,
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)

    constraint = nll.get_constraints()

    opts = trust_region.Trust_Region_Options(max_iter=300)

    result = trust_region.trust_region(
        func,
        grad,
        beta0 if n > 10 else coeff,
        *constraint,
        opts,
    )
    beta_mle = result.x
    abserr_mle = difference.absolute(coeff, beta_mle)
    print("result.success: ", result.success)
    print("coeff: ", coeff)
    print("mle:   ", beta_mle)
    print("abserr_mle: ", abserr_mle)
    assert result.success
    assert 5 < result.iter < 200
    assert abserr_mle < 0.05


class Test_1:
    def test_1(_) -> None:
        run_once(numpy.array([0.5, 0.01, 0.25, 0.7]), 1000, 30)


if __name__ == "__main__":
    Test_1().test_1()
