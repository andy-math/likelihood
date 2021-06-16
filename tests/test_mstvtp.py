# -*- coding: utf-8 -*-
import math

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Copy import Copy
from likelihood.stages.Iterize import Iterize
from likelihood.stages.Linear import Linear
from likelihood.stages.Logistic import Logistic
from likelihood.stages.Merge import Merge
from likelihood.stages.MS_TVTP import MS_TVTP, providers
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region

from tests.common import nll2func


def normpdf(err: float) -> float:
    return 1.0 / math.sqrt(2.0 * math.pi) * math.exp(-err * err / 2.0)


def generate(coeff: ndarray, n: int, seed: int = 0) -> ndarray:
    numpy.random.seed(seed)
    p11b1, p22b1 = coeff
    p11, p22 = 1.0 / (math.exp(-p11b1) + 1.0), 1.0 / (math.exp(-p22b1) + 1.0)
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
        (x, numpy.zeros((n, 1)), numpy.ones((n, 1)), numpy.zeros((n, 10))), axis=1
    )
    beta0 = numpy.array([0.0, 0.0])

    stage1 = Linear(("p11b1",), (2,), 3)
    stage2 = Linear(("p22b1",), (2,), 4)
    stage3 = Merge([Logistic((3,), (3,)), Logistic((4,), (4,))])
    stage4 = Copy((0, 1, 2), (5, 6, 7))
    stage5 = Copy((0, 2), (8, 9))
    stage6 = Copy((2,), (10,))
    submodel1 = Iterize((5, 6, 7), (5, 6, 7))
    submodel2 = Iterize((8, 9, 10), (8, 9, 10))
    stage7 = MS_TVTP(
        (submodel1, submodel2), (), providers["normpdf"], (3, 4), (0, 11, 12)
    )

    nll = likelihood.negLikelihood(
        [stage1, stage2, stage3, stage4, stage5, stage6, stage7],
        None,
        nvars=13,
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)

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
    relerr_mle = difference.relative(coeff, beta_mle)
    print("result.success: ", result.success)
    print("coeff: ", coeff)
    print("mle:   ", beta_mle)
    print("relerr_mle: ", relerr_mle)
    assert result.success
    assert 5 < result.iter < 200
    assert relerr_mle < 0.6


class Test_1:
    def test_1(_) -> None:
        run_once(numpy.array([1.0, 1.0]), 3000)


if __name__ == "__main__":
    Test_1().test_1()
