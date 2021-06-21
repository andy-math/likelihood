# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.KnownIssue import KnownIssue
from likelihood.stages.LogNormpdf import LogNormpdf
from likelihood.stages.Midas_beta import Midas_beta
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region
from overloads.shortcuts import assertNoInfNaN

from tests.common import nll2func


def generate(coeff: ndarray, n: int, k: int, seed: int = 0) -> Tuple[ndarray, ndarray]:
    numpy.random.seed(seed)
    omega1, omega2 = coeff
    assert 1 < omega1 and 1 < omega2
    assert n > k
    kk = numpy.arange(1.0, k + 1.0)[::-1] / k
    kernel = kk ** (omega1 - 1) * (1 - kk) ** (omega2 - 1)
    kernel = kernel / numpy.sum(kernel)
    assertNoInfNaN(kernel)
    x = numpy.zeros((n + k * 10, 1))
    for i in range(k):
        x[i] = numpy.random.randn()
    for i in range(k, n + k * 10):
        start = i - k
        stop = i
        x[i] = x[start:stop, 0] @ kernel + numpy.random.randn()
    start = k * 10
    return x[start:], kernel


def run_once(coeff: ndarray, n: int, k: int, seed: int = 0) -> None:
    x, kkkk = generate(coeff, n, k, seed=seed)
    x, y = x[:-1, :], x[1:, :]
    input = numpy.concatenate((y, x), axis=1)
    beta0 = numpy.array([2.0, 2.0, 1.0])

    stage1 = Midas_beta(("omega1", "omega2"), ("X",), ("X",), k=k)
    stage2 = LogNormpdf("var", ("Y", "X"), ("Y", "X"))

    kernel_reldiff = difference.relative(stage1.kernel(coeff)[0], kkkk[::-1])
    assert kernel_reldiff < 1e-8

    nll = likelihood.negLikelihood(
        ("omega1", "omega2", "var"),
        ("Y", "X"),
        (stage1, stage2),
        None,
        nvars=2,
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)

    constraint = nll.get_constraints()

    opts = trust_region.Trust_Region_Options(max_iter=300)
    opts.tol_grad = 1e-5

    result = trust_region.trust_region(
        func,
        grad,
        beta0,
        *constraint,
        opts,
    )
    beta_mle = result.x[:-1]
    abserr_mle = difference.absolute(coeff, beta_mle)
    print("result.success: ", result.success)
    print("result.delta: ", result.delta)
    print("coeff: ", coeff)
    print("mle:   ", beta_mle)
    print("abserr_mle: ", abserr_mle)
    assert result.success
    assert 2 < result.iter < 200
    assert abserr_mle < 1.05  # (?)


def known_issue(coeff: ndarray, n: int, k: int, seed: int = 0) -> None:
    x, _ = generate(coeff, n, k, seed=seed)
    x, y = x[:-1, :], x[1:, :]
    input = numpy.concatenate((y, x), axis=1)
    beta0 = numpy.array([1.0e308, 1.0e308, 1.0])

    stage1 = Midas_beta(("omega1", "omega2"), ("X",), ("X",), k=k)
    stage2 = LogNormpdf("var", ("Y", "X"), ("Y", "X"))

    ce: Optional[BaseException] = None
    try:
        nll = likelihood.negLikelihood(
            ("omega1", "omega2", "var"), ("Y", "X"), (stage1, stage2), None, nvars=2
        )
        nll.eval(beta0, input, regularize=False)
    except BaseException as e:
        ce = e
    assert isinstance(ce, KnownIssue)
    assert ce.args[0] == "Midas_beta: 权重全为0"


class Test_1:
    def test_1(self) -> None:
        run_once(numpy.array([3.0, 3.0]), 1000, 30)

    def test_2(self) -> None:
        run_once(numpy.array([5.0, 1.5]), 1000, 30)

    def test_3(self) -> None:
        run_once(numpy.array([1.5, 1.5]), 1000, 30)

    def test_4(self) -> None:
        run_once(numpy.array([3.0, 3.0]), 1000, 7)

    def test_5(self) -> None:
        run_once(numpy.array([5.0, 5.0]), 1000, 7)

    def test_6(self) -> None:
        run_once(numpy.array([1.5, 1.5]), 1000, 7)

    def test_7(self) -> None:
        known_issue(numpy.array([2.0, 2.0]), 1000, 7)


if __name__ == "__main__":
    Test_1().test_1()
    Test_1().test_2()
    Test_1().test_3()
    Test_1().test_4()
    Test_1().test_5()
    Test_1().test_6()
    Test_1().test_7()
