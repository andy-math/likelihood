# -*- coding: utf-8 -*-
import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Linear import Linear
from likelihood.stages.LogNormpdf import LogNormpdf
from likelihood.stages.Residual import Residual
from likelihood.Variables import Variables
from optimizer import trust_region
from overloads import difference
from overloads.typedefs import ndarray

from tests.common import nll2func


def run_once(n: int, m: int, seed: int = 0) -> None:
    numpy.random.seed(seed)
    x: ndarray = numpy.concatenate(  # type: ignore
        (numpy.random.randn(n, m), numpy.ones((n, 1))), axis=1
    )
    beta = numpy.random.randn(m + 1)
    y: ndarray = x @ beta + numpy.random.randn(n)
    beta_decomp, _, _, _ = numpy.linalg.lstsq(x, y, rcond=None)  # type: ignore
    abserr_decomp = difference.absolute(beta, beta_decomp)

    nll = likelihood.negLikelihood(
        ("b1", "b2", "b3", "b4", "b5", "b0", "var"),
        ("Y", "var1", "var2", "var3", "var4", "var5", "ones", "zeros"),
        (
            Linear(
                ("b1", "b2", "b3", "b4", "b5", "b0"),
                ("var1", "var2", "var3", "var4", "var5", "ones"),
                ("var1"),
            ),
            Residual(("Y", "var1"), "var1"),
            LogNormpdf("var", ("var1", "zeros"), ("Y", "var1")),
        ),
        None,
    )

    beta0 = numpy.zeros((beta.shape[0] + 1,))
    beta0[-1] = 1.0
    input = Variables(
        tuple(range(n)),
        ("Y", y),
        ("var1", x[:, 0]),
        ("var2", x[:, 1]),
        ("var3", x[:, 2]),
        ("var4", x[:, 3]),
        ("var5", x[:, 4]),
        ("ones", numpy.ones((n,))),
        ("zeros", None),
    )

    func, grad = nll2func(nll, beta0, input, regularize=False)

    opts = trust_region.Trust_Region_Options(max_iter=300)

    constraint = nll.get_constraints()

    result = trust_region.trust_region(
        func,
        grad,
        beta0,
        constraint,
        opts,
    )
    beta_mle = result.x[:-1]
    abserr_mle = difference.absolute(beta, beta_mle)
    print("result.success: ", result.success)
    print("beta:   ", beta)
    print("decomp: ", beta_decomp)
    print("mle:    ", beta_mle)
    print("abserr_decomp: ", abserr_decomp)
    print("abserr_mle:    ", abserr_mle)
    assert result.success
    assert 5 < result.iter < 200
    assert abserr_decomp < 0.1
    assert abserr_mle < 2 * abserr_decomp


class Test_1:
    def test_1(self) -> None:
        run_once(1000, 5)


if __name__ == "__main__":
    Test_1().test_1()
