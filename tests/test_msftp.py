# -*- coding: utf-8 -*-
import math

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Assign import Assign
from likelihood.stages.Copy import Copy
from likelihood.stages.Iterize import Iterize
from likelihood.stages.MS_TVTP import MS_TVTP, providers
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region

from tests.common import nll2func


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
        (x, numpy.zeros((n, 1)), numpy.ones((n, 1)), numpy.zeros((n, 8))), axis=1
    )
    beta0 = numpy.array([0.8, 0.8])

    using_var_names = (
        *("Y", "zeros", "ones"),
        *("Y1", "mean1", "var1"),
        *("Y2", "mean2", "var2"),
        *("p11col", "p22col"),
    )

    stage4 = Copy(("Y", "zeros", "ones"), ("Y1", "mean1", "var1"), (0, 1, 2), (3, 4, 5))
    stage5 = Copy(("Y", "ones"), ("Y2", "mean2"), (0, 2), (6, 7))
    stage6 = Copy(("ones",), ("var2",), (2,), (8,))
    submodel1 = Iterize(
        ("Y1", "mean1", "var1"), ("Y1", "mean1", "var1"), (3, 4, 5), (3, 4, 5)
    )
    submodel2 = Iterize(
        ("Y2", "mean2", "var2"), ("Y2", "mean2", "var2"), (6, 7, 8), (6, 7, 8)
    )
    assign1 = Assign("p11", "p11col", 9, 0.0, 1.0)
    assign2 = Assign("p22", "p22col", 10, 0.0, 1.0)
    stage7 = MS_TVTP(
        (submodel1, submodel2),
        (),
        providers["normpdf"],
        ("p11col", "p22col"),
        ("Y", "p11col", "p22col"),
        (9, 10),
        (0, 9, 10),
    )

    nll = likelihood.negLikelihood(
        ("p11", "p22"),
        using_var_names,
        (stage4, stage5, stage6, assign1, assign2, stage7),
        None,
        nvars=11,
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
    relerr_mle = difference.relative(coeff, beta_mle)
    print("result.success: ", result.success)
    print("coeff: ", coeff)
    print("mle:   ", beta_mle)
    print("relerr_mle: ", relerr_mle)
    assert result.success
    assert 5 < result.iter < 200
    assert relerr_mle < 0.15


class Test_1:
    def test_1(_) -> None:
        run_once(numpy.array([0.8, 0.8]), 1000)


if __name__ == "__main__":
    Test_1().test_1()
