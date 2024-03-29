# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.KnownIssue import KnownIssue
from likelihood.stages.LogNormpdf import LogNormpdf
from likelihood.stages.Midas_exp_group import Midas_exp_group
from likelihood.Variables import Variables
from optimizer import trust_region
from overloads import difference
from overloads.shortcuts import assertNoInfNaN
from overloads.typedefs import ndarray

from tests.common import nll2func


def generate(coeff: ndarray, n: int, k: int, seed: int = 0) -> Tuple[ndarray, ndarray]:
    numpy.random.seed(seed)
    omega = float(coeff)
    assert 0 < omega < 1
    assert n > k
    kernel = omega ** numpy.arange(1.0, k + 1.0)
    kernel = kernel / numpy.sum(kernel)
    assertNoInfNaN(kernel)
    x = numpy.random.randn(n, k)
    return x, x @ kernel + numpy.random.randn(n)


def run_once(coeff: ndarray, n: int, k: int, seed: int = 0) -> None:
    x, y = generate(coeff, n, k, seed=seed)
    input = Variables(
        tuple(range(n)),
        ("Y", y),
        ("X", None),
        *((f"X{i}", x[:, i]) for i in range(k)),
    )
    beta0 = numpy.array([0.5, 1.0])

    nll = likelihood.negLikelihood(
        ("omega", "var"),
        ("Y", "X", *(f"X{i}" for i in range(k))),
        (
            Midas_exp_group("omega", tuple(f"X{i}" for i in range(k)), "X"),
            LogNormpdf("var", ("Y", "X"), ("Y", "X")),
        ),
        None,
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)

    opts = trust_region.Trust_Region_Options(max_iter=300)

    result = trust_region.trust_region(
        func,
        grad,
        beta0,
        nll.get_constraints(),
        opts,
    )
    beta_mle = result.x[:-1]
    abserr_mle = difference.absolute(coeff, beta_mle)
    print("result.success: ", result.success)
    print("coeff: ", coeff)
    print("mle:   ", beta_mle)
    print("abserr_mle: ", abserr_mle)
    assert result.success
    assert 2 < result.iter < 200
    assert abserr_mle < 0.1


def known_issue(coeff: ndarray, n: int, k: int, seed: int = 0) -> None:
    x, y = generate(coeff, n, k, seed=seed)
    input = Variables(
        tuple(range(n)),
        ("Y", y),
        ("X", None),
        *((f"X{i}", x[:, i]) for i in range(k)),
    )
    beta0 = numpy.array([0.0, 1.0])

    stage1 = Midas_exp_group("omega", tuple(f"X{i}" for i in range(k)), "X")
    stage2 = LogNormpdf("var", ("Y", "X"), ("Y", "X"))

    ce: Optional[BaseException] = None
    try:
        nll = likelihood.negLikelihood(
            ("omega", "var"),
            ("Y", "X", *(f"X{i}" for i in range(k))),
            (stage1, stage2),
            None,
        )
        nll.eval(beta0, input, regularize=False)
    except BaseException as e:
        ce = e
    assert isinstance(ce, KnownIssue)
    assert ce.args[0] == "Midas_exp: 权重全为0"


class Test_1:
    def test_1(self) -> None:
        run_once(numpy.array([0.8]), 1000, 30)

    def test_2(self) -> None:
        run_once(numpy.array([0.2]), 1000, 7)

    def test_3(self) -> None:
        known_issue(numpy.array([0.5]), 1000, 7)


if __name__ == "__main__":
    Test_1().test_1()
    Test_1().test_2()
    Test_1().test_3()
