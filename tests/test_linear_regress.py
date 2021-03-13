# -*- coding: utf-8 -*-
import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Linear import Linear
from likelihood.stages.LogNormpdf import LogNormpdf
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region


def run_once(n: int, m: int, seed: int = 0) -> None:
    numpy.random.seed(seed)
    x = numpy.concatenate((numpy.random.randn(n, m), numpy.ones((n, 1))), axis=1)
    beta = numpy.random.randn(m + 1)
    y = x @ beta + numpy.random.randn(n)
    beta_decomp, _, _, _ = numpy.linalg.lstsq(x, y, rcond=None)  # type: ignore
    abserr_decomp = difference.absolute(beta, beta_decomp)

    stage1 = Linear(["b1", "b2", "b3", "b4", "b5", "b0"], list(range(1, 7)), [1])
    stage2 = LogNormpdf("var", (0, 1), 0)
    nll = likelihood.negLikelihood([stage1, stage2], 7)

    beta0 = numpy.zeros((beta.shape[0] + 1,))
    beta0[-1] = 1.0
    input = numpy.concatenate((y.reshape((-1, 1)), x), axis=1)

    assert nll.eval(beta0, input) == nll.eval(beta0, input)

    def func(x: ndarray) -> float:
        return nll.eval(x, input)

    def grad(x: ndarray) -> ndarray:
        return nll.grad(x, input)

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
    abserr_mle = difference.absolute(beta, beta_mle)
    print("result.success: ", result.success)
    print("beta:   ", beta)
    print("decomp: ", beta_decomp)
    print("mle:    ", beta_mle)
    print("abserr_decomp: ", abserr_decomp)
    print("abserr_mle:    ", abserr_mle)
    assert result.success
    assert 5 < result.iter < 20
    assert abserr_decomp < 0.1
    assert abserr_mle < 2 * abserr_decomp


class Test_1:
    def test_1(self) -> None:
        run_once(1000, 5)


if __name__ == "__main__":
    Test_1().test_1()
