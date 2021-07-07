# -*- coding: utf-8 -*-
import math

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Garch_mean import Garch_mean
from likelihood.stages.LogNormpdf_var import LogNormpdf_var
from likelihood.Variables import Variables
from overloads import difference
from overloads.typing import ndarray
from optimizer import trust_region

from tests.common import nll2func


def generate(coeff: ndarray, n: int, seed: int = 0) -> ndarray:
    numpy.random.seed(seed)
    c, a, b = coeff
    var = c / (1 - a - b)
    x = numpy.zeros((n,))
    x[0] = numpy.random.normal(loc=0, scale=math.sqrt(var), size=1)
    for i in range(1, n):
        var = c + a * x[i - 1] * x[i - 1] + b * var
        x[i] = numpy.random.normal(loc=0, scale=math.sqrt(var), size=1)
    return x


def run_once(coeff: ndarray, n: int, seed: int = 0) -> None:
    x = generate(coeff, n, seed=seed)
    x, y = x[:-1], x[1:]

    input = Variables(
        tuple(range(n - 1)), ("Y", y), ("mean", None), ("var", None), ("EX2", None)
    )
    beta0 = numpy.array([numpy.std(y) ** 2 * 0.1, 0.1, 0.8])

    stage1 = Garch_mean(("c", "a", "b"), ("Y", "mean"), ("Y", "mean", "var", "EX2"))
    stage2 = LogNormpdf_var(("Y", "var"), ("Y", "var"))

    nll = likelihood.negLikelihood(
        ("c", "a", "b"), ("Y", "mean", "var", "EX2"), (stage1, stage2), None
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)

    constraint = nll.get_constraints()

    opts = trust_region.Trust_Region_Options(max_iter=300)
    # opts.check_iter = 50

    result = trust_region.trust_region(
        func,
        grad,
        beta0 if n > 10 else coeff,
        constraint,
        opts,
    )
    beta_mle = result.x
    abserr_mle = difference.absolute(coeff, beta_mle)
    print("result.success: ", result.success)
    print("coeff: ", coeff)
    print("mle:   ", beta_mle)
    print("abserr_mle: ", abserr_mle)
    print("result.grad: ", result.gradient)
    assert result.success
    assert 5 < result.iter < 200
    assert abserr_mle < 0.05


class Test_1:
    def test_1(_) -> None:
        run_once(numpy.array([0.01, 0.25, 0.7]), 1000)


if __name__ == "__main__":
    Test_1().test_1()
