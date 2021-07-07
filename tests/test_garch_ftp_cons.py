# -*- coding: utf-8 -*-
import math

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Assign import Assign
from likelihood.stages.Copy import Copy
from likelihood.stages.Garch_mean import Garch_mean
from likelihood.stages.Mapping import Mapping
from likelihood.stages.MS_TVTP import MS_TVTP, providers
from likelihood.Variables import Variables
from overloads.typing import ndarray
from optimizer import trust_region
from overloads import difference

from tests.common import nll2func


def normpdf(err: float, var: float) -> float:
    return 1.0 / math.sqrt(2.0 * math.pi * var) * math.exp(-(err * err) / (2.0 * var))


def generate(coeff: ndarray, n: int, seed: int = 0) -> ndarray:
    numpy.random.seed(seed)
    p11, p22, c1, c2, a, b = coeff
    p1, p2 = 0.5, 0.5
    var1 = c1 / (1.0 - a - b)
    var2 = c2 / (1.0 - a - b)
    x = numpy.zeros((n,))
    for i in range(n):
        path11, path22 = p1 * p11, p2 * p22
        p1, p2 = path11 + p2 * (1 - p22), p1 * (1 - p11) + path22
        contrib11, contrib22 = path11 / p1, path22 / p2
        var1, var2 = (
            contrib11 * var1 + (1 - contrib11) * var2,
            (1 - contrib22) * var1 + contrib22 * var2,
        )

        x[i] = p1 * numpy.random.normal(
            loc=0, scale=math.sqrt(var1), size=1
        ) + p2 * numpy.random.normal(loc=0, scale=math.sqrt(var2), size=1)

        f1, f2 = normpdf(x[i], var1), normpdf(x[i], var2)

        p1, p2 = p1 * f1, p2 * f2
        p1, p2 = p1 / (p1 + p2), p2 / (p1 + p2)

        var1 = c1 + a * x[i] * x[i] + b * var1
        var2 = c2 + a * x[i] * x[i] + b * var2

    return x


def run_once(coeff: ndarray, n: int, seed: int = 0) -> None:
    x = generate(coeff, n, seed=seed)

    input = Variables(
        tuple(range(n)),
        *(("Y", x), ("zeros", None), ("ones", numpy.ones((n,)))),
        *(("Y1", None), ("mean1", None), ("var1", None), ("EX2_1", None)),
        *(("Y2", None), ("mean2", None), ("var2", None), ("EX2_2", None)),
        *(("p11col", None), ("p22col", None)),
    )
    beta0 = numpy.array([0.8, 0.8, 0.011, 0.022, 0.078, 0.89])

    using_var_names = (
        *("Y", "zeros", "ones"),
        *("Y1", "mean1", "var1", "EX2_1"),
        *("Y2", "mean2", "var2", "EX2_2"),
        *("p11col", "p22col"),
    )

    nll = likelihood.negLikelihood(
        ("p11", "p22", "c1", "c2", "a", "b"),
        using_var_names,
        (
            Copy(("Y", "zeros"), ("Y1", "mean1")),
            Copy(("Y", "zeros"), ("Y2", "mean2")),
            Assign("p11", "p11col", 0.0, 1.0),
            Assign("p22", "p22col", 0.0, 1.0),
            Mapping(
                {
                    "c1": ("c1",),
                    "c2": ("c2",),
                    "a": ("a1", "a2"),
                    "b": ("b1", "b2"),
                },
                MS_TVTP(
                    (
                        Garch_mean(
                            ("c1", "a1", "b1"),
                            ("Y1", "mean1"),
                            ("Y1", "mean1", "var1", "EX2_1"),
                        ),
                        Garch_mean(
                            ("c2", "a2", "b2"),
                            ("Y2", "mean2"),
                            ("Y2", "mean2", "var2", "EX2_2"),
                        ),
                    ),
                    providers["normpdf"],
                    ("p11col", "p22col"),
                    ("Y", "zeros", "ones", "p11col", "p22col"),
                ),
            ),
        ),
        None,
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)

    constraint = nll.get_constraints()

    opts = trust_region.Trust_Region_Options(max_iter=99999)
    opts.check_iter = 25
    opts.tol_grad = 1e-4
    opts.border_abstol = 1e-10

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
    print("mle:   ", [round(x, 6) for x in beta_mle])
    print("abserr_mle: ", abserr_mle)
    # assert result.success
    assert 5 < result.iter < 3000


class Test_1:
    def test_1(_) -> None:
        run_once(numpy.array([0.8, 0.8, 0.011, 0.022, 0.078, 0.89]), 1000)


if __name__ == "__main__":
    Test_1().test_1()
