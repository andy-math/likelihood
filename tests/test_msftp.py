# -*- coding: utf-8 -*-
import math

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Copy import Copy
from likelihood.stages.Iterize import Iterize
from likelihood.stages.LogNormpdf import LogNormpdf
from likelihood.stages.MS_FTP import MS_FTP
from likelihood.stages.MS_TVTP import providers
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region


def normpdf(err: float) -> float:
    return 1.0 / math.sqrt(2.0 * math.pi) * math.exp(-err * err / 2.0)


def generate(coeff: ndarray, n: int, seed: int = 0) -> ndarray:
    numpy.random.seed(seed)
    p11, p22 = coeff
    p1, p2 = 0.5, 0.5
    x = numpy.zeros((n, 1))
    for i in range(n):
        p1, p2 = p1 * p11 + p2 * (1 - p22), p1 * (1 - p11) + p2 * p22
        x[i] = p1 * numpy.random.normal(
            loc=0, scale=1, size=1
        ) + p2 * numpy.random.normal(loc=1, scale=1, size=1)
        f1, f2 = normpdf(x[i] - 0), normpdf(x[i] - 1)
        p1, p2 = p1 * f1, p2 * f2
        p1, p2 = p1 / (p1 + p2), p2 / (p1 + p2)
    return x


def run_once(coeff: ndarray, n: int, seed: int = 0) -> None:
    x = generate(coeff, n, seed=seed)

    input = numpy.concatenate(
        (x, numpy.zeros((n, 1)), numpy.ones((n, 1)), numpy.zeros((n, 13))), axis=1
    )
    beta0 = numpy.array([0.8, 0.8, 1.0])

    stage4 = Copy((0, 1, 2), (5, 6, 7))
    stage5 = Copy((0, 2), (8, 9))
    stage6 = Copy((2,), (10,))
    submodel1 = Iterize((5, 6, 7), (5, 6, 7))
    submodel2 = Iterize((8, 9, 10), (8, 9, 10))
    stage7 = MS_FTP(
        ("p11", "p22"),
        (submodel1, submodel2),
        [],
        providers["normpdf"],
        (11, 12, 13, 14, 15),
    )
    stage8 = LogNormpdf("var", (0, 12), (0, 1))

    nll = likelihood.negLikelihood(
        [stage4, stage5, stage6, stage7, stage8],
        None,
        nvars=16,
    )

    assert (
        nll.eval(beta0, input, regularize=False, debug=True)[0]
        == nll.eval(beta0, input, regularize=False, debug=True)[0]
    )
    assert numpy.all(
        nll.grad(beta0, input, regularize=False, debug=True)
        == nll.grad(beta0, input, regularize=False, debug=True)
    )

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
    beta_mle = result.x[:-1]
    relerr_mle = difference.relative(coeff, beta_mle)
    print("result.success: ", result.success)
    print("coeff: ", coeff)
    print("mle:   ", beta_mle)
    print("relerr_mle: ", relerr_mle)
    assert result.success
    assert 5 < result.iter < 200
    assert relerr_mle < 0.05


class Test_1:
    def test_1(_) -> None:
        run_once(numpy.array([0.8, 0.8]), 1000)


if __name__ == "__main__":
    Test_1().test_1()