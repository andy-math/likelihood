# -*- coding: utf-8 -*-
import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Lasso import Lasso
from likelihood.stages.Linear import Linear
from likelihood.stages.LogNormpdf import LogNormpdf
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region


def run_once(n: int, m: int, seed: int = 0) -> None:
    numpy.random.seed(seed)
    rrrr = numpy.random.randn(n, m)
    x = numpy.concatenate((rrrr, rrrr, numpy.ones((n, 1))), axis=1)
    beta = numpy.random.randn(2 * m + 1)
    y = x @ beta + numpy.random.randn(n)
    beta_decomp, _, _, _ = numpy.linalg.lstsq(x, y, rcond=None)  # type: ignore
    abserr_decomp = difference.absolute(
        beta[:m] + beta[m:-1], beta_decomp[:m] + beta_decomp[m:-1]
    )

    stage1 = Linear(
        [*[f"b{i}" for i in range(1, 2 * m + 1)], "b0"], list(range(1, 2 * m + 2)), [1]
    )
    stage2 = LogNormpdf("var", (0, 1), (0, 1))
    penalty = Lasso(1.0, (0, 1), 0)
    nll = likelihood.negLikelihood([stage1, stage2], penalty, nvars=2 * m + 2)

    beta0 = numpy.zeros((beta.shape[0] + 1,))
    beta0[-1] = 1.0
    input = numpy.concatenate((y.reshape((-1, 1)), x), axis=1)

    assert nll.eval(beta0, input, regularize=True) == nll.eval(
        beta0, input, regularize=True
    )

    def func(x: ndarray) -> float:
        return nll.eval(x, input, regularize=True)

    def grad(x: ndarray) -> ndarray:
        return nll.grad(x, input, regularize=True)

    opts = trust_region.Trust_Region_Options(max_iter=300)

    constraint = nll.get_constraint()

    result = trust_region.trust_region(
        func,
        grad,
        beta0,
        *constraint,
        opts,
    )
    beta_mle = result.x[:-1]
    abserr_mle = difference.absolute(
        beta[:m] + beta[m:-1], beta_mle[:m] + beta_mle[m:-1]
    )
    print("result.success: ", result.success)
    print("beta:   ", beta[:-1])
    print("decomp: ", beta_decomp[:-1])
    print("mle:    ", beta_mle[:-1])
    print("abserr_decomp: ", abserr_decomp)
    print("abserr_mle:    ", abserr_mle)
    return
    assert result.success
    assert 5 < result.iter < 20
    assert abserr_decomp < 0.1
    assert abserr_mle < 2 * abserr_decomp


class Test_1:
    def test_1(self) -> None:
        run_once(1000, 3)


if __name__ == "__main__":
    Test_1().test_1()
