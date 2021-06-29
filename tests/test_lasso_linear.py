# -*- coding: utf-8 -*-
import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Lasso import Lasso
from likelihood.stages.Linear import Linear
from likelihood.stages.LogNormpdf import LogNormpdf
from likelihood.Variables import Variables
from overloads import difference
from optimizer import trust_region

from tests.common import nll2func


def run_once(n: int, m: int, seed: int = 0) -> None:
    numpy.random.seed(seed)
    rrrr = numpy.random.randn(n, m)
    rrrr[:, -1] = rrrr[:, 0] - rrrr[:, -1] / 1000
    x = rrrr
    beta = n * numpy.random.randn(m)
    beta[-1] = 0.0
    y = x @ beta + numpy.random.randn(n)
    beta_decomp, _, _, _ = numpy.linalg.lstsq(x, y, rcond=None)  # type: ignore
    relerr_decomp = difference.relative(beta[:-1], beta_decomp[:-1])

    stage1 = Linear(
        tuple(f"b{i}" for i in range(1, m + 1)),
        tuple(f"var{i}" for i in range(1, m + 1)),
        "var1",
    )
    stage2 = LogNormpdf("var", ("Y", "var1"), ("Y", "var1"))
    penalty = Lasso(stage1.coeff_names, 1.0, ("Y", "var1"), "Y")
    nll = likelihood.negLikelihood(
        stage1.coeff_names + ("var",),
        ("Y",) + tuple(f"var{i}" for i in range(1, m + 1)),
        (stage1, stage2),
        penalty,
    )

    beta0 = numpy.zeros((beta.shape[0] + 1))
    beta0[-1] = 1.0
    input = Variables(
        tuple(range(n)), ("Y", y), *((f"var{i+1}", x[:, i]) for i in range(m))
    )

    func, grad = nll2func(nll, beta0, input, regularize=True)

    opts = trust_region.Trust_Region_Options(max_iter=99999)

    constraint = nll.get_constraints()

    result = trust_region.trust_region(
        func,
        grad,
        beta0,
        constraint,
        opts,
    )
    beta_mle = result.x[:-1]
    relerr_mle = difference.relative(beta[:-1], beta_mle[:-1])
    print("result.success: ", result.success)
    print("result.delta: ", result.delta)
    print("beta:   ", beta)
    print("decomp: ", beta_decomp)
    print("mle:    ", beta_mle)
    print("relerr_decomp: ", relerr_decomp)
    print("relerr_mle:    ", relerr_mle)
    assert result.success
    assert 5 < result.iter < 1000
    assert relerr_decomp < 0.1
    assert relerr_mle < relerr_decomp


class Test_1:
    def test_1(self) -> None:
        run_once(1000, 4)


if __name__ == "__main__":
    Test_1().test_1()
