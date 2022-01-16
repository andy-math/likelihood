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
from likelihood.Variables import Variables
from optimizer import trust_region
from overloads import difference
from overloads.typedefs import ndarray

from tests.common import nll2func


def normpdf(err: float) -> float:
    return 1.0 / math.sqrt(2.0 * math.pi) * math.exp(-err * err / 2.0)


def generate(coeff: ndarray, n: int, seed: int = 0) -> ndarray:
    numpy.random.seed(seed)
    p11b1, p22b1 = coeff
    p11, p22 = 1.0 / (math.exp(-p11b1) + 1.0), 1.0 / (math.exp(-p22b1) + 1.0)
    p1, p2 = 0.5, 0.5
    x = numpy.zeros((n,))
    for i in range(n):
        p1, p2 = p1 * p11 + p2 * (1 - p22), p1 * (1 - p11) + p2 * p22
        x[i] = float(p1 * numpy.random.normal(loc=0, scale=1, size=1)) + float(
            p2 * numpy.random.normal(loc=1, scale=1, size=1)
        )
        f1, f2 = normpdf(x[i] - 0), normpdf(x[i] - 1)
        p1, p2 = p1 * f1, p2 * f2
        p1, p2 = p1 / (p1 + p2), p2 / (p1 + p2)
    return x


class NLL(likelihood.negLikelihood):
    def __init__(self) -> None:
        super().__init__(
            ("p11b1", "p22b1"),
            (
                ("Y", "zeros", "ones")
                + ("Y1", "mean1", "var1")
                + ("Y2", "mean2", "var2")
                + ("p11col", "p22col")
            ),
            (
                Linear(("p11b1",), ("ones",), "p11col"),
                Linear(("p22b1",), ("ones",), "p22col"),
                Merge(
                    (
                        Logistic(("p11col",), ("p11col",)),
                        Logistic(("p22col",), ("p22col",)),
                    )
                ),
                Copy(("Y", "zeros", "ones"), ("Y1", "mean1", "var1")),
                Copy(("Y", "ones"), ("Y2", "mean2")),
                Copy(("ones",), ("var2",)),
                MS_TVTP(
                    (
                        Iterize(("Y1", "mean1", "var1"), ("Y1", "mean1", "var1")),
                        Iterize(("Y2", "mean2", "var2"), ("Y2", "mean2", "var2")),
                    ),
                    providers["normpdf"],
                    ("p11col", "p22col"),
                    ("Y", "zeros", "ones", "p11col", "p22col"),
                ),
            ),
            None,
        )


def run_once(coeff: ndarray, n: int, seed: int = 0) -> None:
    x = generate(coeff, n, seed=seed)

    input = Variables(
        tuple(range(n)),
        *(("Y", x), ("zeros", None), ("ones", numpy.ones((n,)))),
        *(("Y1", None), ("mean1", None), ("var1", None)),
        *(("Y2", None), ("mean2", None), ("var2", None)),
        *(("p11col", None), ("p22col", None)),
    )
    beta0 = numpy.array([0.0, 0.0])

    nll = NLL()
    func, grad = nll2func(nll, beta0, input, regularize=False)

    opts = trust_region.Trust_Region_Options(max_iter=300)

    result = trust_region.trust_region(
        func,
        grad,
        beta0 if n > 10 else coeff,
        nll.get_constraints(),
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
    def test_1(self) -> None:
        run_once(numpy.array([1.0, 1.0]), 1000)


if __name__ == "__main__":
    Test_1().test_1()
