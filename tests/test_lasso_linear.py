# -*- coding: utf-8 -*-
from typing import cast

import numpy
import numpy.linalg
import scipy.stats  # type: ignore

from likelihood import likelihood
from likelihood.stages.Lasso import Lasso
from likelihood.stages.Linear import Linear
from likelihood.stages.LogNormpdf import LogNormpdf
from likelihood.Variables import Variables
from optimizer import trust_region
from overloads import difference
from overloads.typedefs import ndarray
from tests.common import nll2func


class Sample:
    beta: ndarray
    X: ndarray
    Y: ndarray
    lambda_: float
    beta_decomp: ndarray

    def symm_eig(self, A: ndarray) -> ndarray:
        A = (A.T + A) / 2
        return cast(ndarray, numpy.linalg.eigh(A)[0])

    def orthogonal_X(self, X: ndarray) -> ndarray:
        for i in range(1, X.shape[1]):
            norm = numpy.sqrt(X[:, i] @ X[:, i])
            X[:, i] -= X[:, :i] @ numpy.linalg.lstsq(X[:, :i], X[:, i], rcond=None)[0]
            X[:, i] *= norm / numpy.sqrt(X[:, i] @ X[:, i])
        return X

    def soft_threshold(self, beta: ndarray, lambda_: ndarray) -> ndarray:
        beta = numpy.sign(beta) * numpy.maximum(numpy.abs(beta) - lambda_, 0.0)
        return beta

    def lasso_decomp(self) -> ndarray:
        m, n = self.X.shape
        beta_decomp: ndarray = numpy.linalg.lstsq(self.X, self.Y, rcond=None)[0]
        beta_decomp = self.soft_threshold(
            beta_decomp,
            (m / 2.0 * self.lambda_)
            * numpy.linalg.lstsq(self.X.T @ self.X, numpy.ones((n,)), rcond=None)[0],
        )
        return beta_decomp

    def __init__(self, m: int, n: int) -> None:
        self.beta = self.symm_eig(numpy.random.rand(n, n).T)
        self.X = self.orthogonal_X(scipy.stats.norm.ppf(numpy.random.rand(n, m).T))
        self.Y = self.X @ self.beta + scipy.stats.norm.ppf(numpy.random.rand(m))
        self.lambda_ = 2 * numpy.quantile(
            numpy.abs(numpy.linalg.lstsq(self.X, self.Y, rcond=None)[0]), 0.3
        )
        self.beta_decomp = self.lasso_decomp()


def run_once(m: int, n: int) -> None:
    sample = Sample(m, n)

    beta0 = numpy.zeros((n + 1))
    beta0[-1] = 1.0
    input = Variables(
        tuple(range(m)),
        ("Y", sample.Y),
        *((f"var{i+1}", sample.X[:, i]) for i in range(n)),
    )

    stage1 = Linear(
        tuple(f"b{i}" for i in range(1, n + 1)),
        tuple(f"var{i}" for i in range(1, n + 1)),
        "var1",
    )
    stage2 = LogNormpdf("var", ("Y", "var1"), ("Y", "var1"))
    penalty = Lasso(stage1.coeff_names, sample.lambda_, ("Y", "var1"), "Y")
    nll = likelihood.negLikelihood(
        stage1.coeff_names + ("var",),
        ("Y",) + tuple(f"var{i}" for i in range(1, n + 1)),
        (stage1, stage2),
        penalty,
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
    abserr = difference.absolute(beta_mle, sample.beta_decomp)
    relerr = abserr / numpy.mean(numpy.abs(sample.beta_decomp))
    print(f"result.success: {result.success}")
    print(f"relerr        : {relerr}")
    print(f"abserr        : {abserr}")
    print(
        "result.x      : \n",
        numpy.concatenate(
            (
                sample.beta_decomp.reshape((-1, 1)),
                beta_mle.reshape((-1, 1)),
            ),
            axis=1,
        ),
    )
    assert relerr < 0.3
    assert abserr < 0.3


class Test_1:
    def test1(self) -> None:
        numpy.random.seed(5489)
        run_once(m=1000, n=4)
        run_once(m=1000, n=8)
        run_once(m=1000, n=16)
        run_once(m=1000, n=32)


if __name__ == "__main__":
    Test_1().test1()
