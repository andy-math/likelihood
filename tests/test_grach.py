# -*- coding: utf-8 -*-
import math

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Garch import Garch
from likelihood.stages.LogNormpdf_var import LogNormpdf_var
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region


def generate(coeff: ndarray, n: int, seed: int = 0) -> ndarray:
    numpy.random.seed(seed)
    c, a, b = coeff
    var = c / (1 - a - b)
    x = numpy.zeros((n, 1))
    x[0] = numpy.random.normal(loc=0, scale=math.sqrt(var), size=1)
    for i in range(1, n):
        var = c + a * x[i - 1] * x[i - 1] + b * var
        x[i] = numpy.random.normal(loc=0, scale=math.sqrt(var), size=1)
    return x


def run_once(coeff: ndarray, n: int, seed: int = 0) -> None:
    x = generate(coeff, n, seed=seed)
    x, y = x[:-1, :], x[1:, :]
    input = numpy.concatenate((y, x), axis=1)
    beta0 = numpy.array([numpy.std(input[:, 0]) ** 2 * 0.1, 0.1, 0.8])

    stage1_cover = Garch(("c", "a", "b"), 1, 1)
    stage1 = Garch(("c", "a", "b"), 1, 1)
    stage2 = LogNormpdf_var((0, 1), (0, 1))

    nll = likelihood.negLikelihood([stage1_cover, stage2], None, nvars=2)
    assert (
        nll.eval(beta0, input, regularize=False)[0]
        == nll.eval(beta0, input, regularize=False)[0]
    )
    assert numpy.all(
        nll.grad(beta0, input, regularize=False)
        == nll.grad(beta0, input, regularize=False)
    )

    nll = likelihood.negLikelihood([stage1, stage2], None, nvars=2)
    assert (
        nll.eval(beta0, input, regularize=False)[0]
        == nll.eval(beta0, input, regularize=False)[0]
    )
    assert numpy.all(
        nll.grad(beta0, input, regularize=False)
        == nll.grad(beta0, input, regularize=False)
    )

    def func(x: ndarray) -> float:
        return nll.eval(x, input, regularize=False)[0]

    def grad(x: ndarray) -> ndarray:
        return nll.grad(x, input, regularize=False)

    constraint = nll.get_constraint()

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
        run_once(numpy.array([0.01, 0.25, 0.7]), 1000)


if __name__ == "__main__":
    Test_1().test_1()
