# -*- coding: utf-8 -*-

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Linear import Linear
from likelihood.stages.Log import Log
from likelihood.stages.LogNormpdf import LogNormpdf
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region

from tests.common import nll2func


def generate(coeff: ndarray, n: int, seed: int = 0) -> ndarray:
    numpy.random.seed(seed)
    x = numpy.concatenate((numpy.random.rand(n, 1) + 0.01, numpy.ones((n, 1))), axis=1)
    y = numpy.log(x @ coeff) + numpy.random.randn(n) / 3
    y = y.reshape((-1, 1))
    return numpy.concatenate((y, x), axis=1)  # type: ignore


def run_once(coeff: ndarray, n: int, seed: int = 0) -> None:
    input = generate(coeff, n, seed=seed)
    beta0 = numpy.array([1.0, 1.0, 1.0])

    stage1 = Linear(("b1", "b0"), ("X", "ones"), "X", (1, 2), 1)
    stage2 = Log("X", "X", 1, 1)
    stage3 = LogNormpdf("var", ("Y", "X"), ("Y", "X"), (0, 1), (0, 1))

    nll = likelihood.negLikelihood(
        ("b1", "b0", "var"), ("Y", "X", "ones"), (stage1, stage2, stage3), None, nvars=3
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)

    constraint = nll.get_constraints()
    constraint[2][:-1] = 0.0

    opts = trust_region.Trust_Region_Options(max_iter=300)
    opts.check_rel = 0.02

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
    assert relerr_mle < 0.1  # (?)


class Test_1:
    def test_1(_) -> None:
        run_once(numpy.array([6.0, 3.0]), 10000)


if __name__ == "__main__":
    Test_1().test_1()
