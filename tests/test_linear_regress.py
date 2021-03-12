# -*- coding: utf-8 -*-
import numpy
import numpy.linalg

# from likelihood import likelihood
from numerical import difference

numpy.random.seed(0)
n = 1000
m = 5


x = numpy.concatenate((numpy.random.randn(n, m), numpy.ones((n, 1))), axis=1)
beta = numpy.random.randn(m + 1)
y = x @ beta + numpy.random.randn(n)
beta_decomp, _, _, _ = numpy.linalg.lstsq(x, y)  # type: ignore
print("beta:   ", beta)
print("decomp: ", beta_decomp)
print("abserr: ", difference.absolute(beta, beta_decomp))


def make_stage() -> None:
    pass  # stage1 = likelihood.Linear()


if __name__ == "__main__":
    pass
