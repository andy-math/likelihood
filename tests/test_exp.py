# -*- coding: utf-8 -*-

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Exp import Exp
from likelihood.stages.Linear import Linear
from likelihood.stages.LogNormpdf import LogNormpdf
from likelihood.Variables import Variables
from optimizer import trust_region
from overloads import difference
from overloads.typedefs import ndarray

from tests.common import nll2func


def generate(coeff: ndarray, n: int, seed: int = 0) -> Variables[int]:
    numpy.random.seed(seed)
    x: ndarray = numpy.concatenate(  # type: ignore
        (numpy.random.rand(n, 1), numpy.ones((n, 1))), axis=1
    )
    y = numpy.exp(x @ coeff) + numpy.random.randn(n)
    return Variables(tuple(range(n)), ("Y", y), ("X", x[:, 0]), ("ones", x[:, 1]))


def run_once(coeff: ndarray, n: int, seed: int = 0) -> None:
    input = generate(coeff, n, seed=seed)
    beta0 = numpy.array([0.0, 0.0, 1.0])

    nll = likelihood.negLikelihood(
        ("b1", "b0", "var"),
        ("Y", "X", "ones"),
        (
            Linear(("b1", "b0"), ("X", "ones"), "X"),
            Exp("X", "X"),
            LogNormpdf("var", ("Y", "X"), ("Y", "X")),
        ),
        None,
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)
    constraint = nll.get_constraints()
    opts = trust_region.Trust_Region_Options(max_iter=300)
    opts.check_rel = 0.02
    result = trust_region.trust_region(
        func,
        grad,
        beta0 if n > 10 else coeff,
        constraint,
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
    def test_1(self) -> None:
        run_once(numpy.array([6.0, -3.0]), 1000)


if __name__ == "__main__":
    Test_1().test_1()
