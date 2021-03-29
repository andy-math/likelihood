# -*- coding: utf-8 -*-
import math

import numpy
import numpy.linalg
from likelihood import likelihood
from likelihood.stages.Garch_mean import Garch_mean
from likelihood.stages.Linear import Linear
from likelihood.stages.Logistic import Logistic
from likelihood.stages.LogNormpdf_var import LogNormpdf_var
from likelihood.stages.MS_TVTP import MS_TVTP, providers
from likelihood.stages.Residual import Residual
from numerical import difference
from numerical.typedefs import ndarray
from optimizer import trust_region


def normpdf(err: float, var: float) -> float:
    return 1.0 / math.sqrt(2.0 * math.pi * var) * math.exp(-(err * err) / (2.0 * var))


def generate(coeff: ndarray, n: int, seed: int = 0) -> ndarray:
    numpy.random.seed(seed)
    p11b1, p22b1, c1, a1, b1, c2, a2, b2 = coeff
    p11, p22 = 1.0 / (math.exp(-p11b1) + 1.0), 1.0 / (math.exp(-p22b1) + 1.0)
    p1, p2 = 0.5, 0.5
    var1 = c1 / (1.0 - a1 - b1)
    var2 = c2 / (1.0 - a2 - b2)
    x = numpy.zeros((n, 1))
    for i in range(n):
        path11, path22 = p1 * p11, p2 * p22
        p1, p2 = path11 + p2 * (1 - p22), p1 * (1 - p11) + path22
        contrib11, contrib22 = path11 / p1, path22 / p2
        var1, var2 = (
            contrib11 * var1 + (1 - contrib11) * var2,
            (1 - contrib22) * var1 + contrib22 * var2,
        )

        x[i] = numpy.random.normal(
            loc=0, scale=math.sqrt(p1 * var1 + p2 * var2), size=1
        )
        f1, f2 = normpdf(x[i], var1), normpdf(x[i], var2)

        p1, p2 = p1 * f1, p2 * f2
        p1, p2 = p1 / (p1 + p2), p2 / (p1 + p2)

        var1 = c1 + a1 * x[i] * x[i] + b1 * var1
        var2 = c2 + a2 * x[i] * x[i] + b2 * var2

    return x


def run_once(coeff: ndarray, n: int, seed: int = 0) -> None:
    x = generate(coeff, n, seed=seed)

    input = numpy.concatenate(
        (x, numpy.zeros((n, 1)), numpy.ones((n, 1)), numpy.zeros((n, 16))), axis=1
    )
    beta0 = numpy.array([1.0, 1.0, 1.0e-6, 0.099, 0.89, 1.0, 1.0e-6, 1.0e-6])

    stage1 = Linear(["p11b1"], (2,), 3)
    stage2 = Linear(["p22b1"], (2,), 4)
    stage3 = Logistic((3, 4), (3, 4))
    # x mu var EX2
    submodel1 = Garch_mean(("c1", "a1", "b1"), (0, 1), (5, 6, 7, 8))
    submodel2 = Garch_mean(("c2", "a2", "b2"), (0, 1), (9, 10, 11, 12))
    stage4 = MS_TVTP(
        (submodel1, submodel2),
        [],
        providers["normpdf"],
        (3, 4),
        (13, 14, 15, 16, 17, 18),
    )
    stage5 = Residual((0, 14), 0)
    stage6 = LogNormpdf_var((0, 15), (0, 15))

    nll = likelihood.negLikelihood(
        [stage1, stage2, stage3, stage4, stage5, stage6], None, nvars=19
    )

    assert (
        nll.eval(beta0, input, regularize=False, debug=True)[0]
        == nll.eval(beta0, input, regularize=False, debug=True)[0]
    )
    assert numpy.all(
        nll.grad(beta0, input, regularize=False, debug=True)
        == nll.grad(beta0, input, regularize=False, debug=True)
    )

    # print(nll.grad(beta0, input, regularize=False, debug=True).reshape((-1, 1)))
    # return

    assert (
        nll.eval(beta0, input, regularize=False)[0]
        == nll.eval(beta0, input, regularize=False)[0]
    )
    assert numpy.all(
        nll.grad(beta0, input, regularize=False)
        == nll.grad(beta0, input, regularize=False)
    )

    def func(x: ndarray) -> float:
        return nll.eval(x, input, regularize=False)[0]

    def grad(x: ndarray) -> ndarray:
        return nll.grad(x, input, regularize=False)

    constraint = nll.get_constraint()
    # 限制转移概率区间
    # constraint[2][:2] = 1.0
    # constraint[3][:2] = 3.0

    opts = trust_region.Trust_Region_Options(max_iter=99999)
    opts.check_iter = 40
    opts.check_rel = 0.05

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
    print("mle:   ", [round(x, 6) for x in beta_mle])
    print("abserr_mle: ", relerr_mle)
    return
    assert result.success
    assert 5 < result.iter < 200
    assert relerr_mle < 0.1


class Test_1:
    def test_1(_) -> None:
        run_once(numpy.array([1.0, 1.0, 0.011, 0.099, 0.89, 1.0, 0.0, 0.0]), 1000)


if __name__ == "__main__":
    Test_1().test_1()