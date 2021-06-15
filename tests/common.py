from typing import Callable, Tuple

import numpy
from likelihood.likelihood import negLikelihood
from numerical.typedefs import ndarray


def nll2func(
    nll: negLikelihood, beta0: ndarray, input: ndarray, *, regularize: bool
) -> Tuple[Callable[[ndarray], float], Callable[[ndarray], ndarray]]:
    (trial1, output1) = nll.eval(beta0, input, regularize=regularize, debug=True)
    (trial2, output2) = nll.eval(beta0, input, regularize=regularize, debug=True)
    assert trial1 == trial2
    assert numpy.all(output1 == output2)

    grad1 = nll.grad(beta0, input, regularize=regularize, debug=True)
    grad2 = nll.grad(beta0, input, regularize=regularize, debug=True)
    assert numpy.all(grad1 == grad2)

    (trial1, output1) = nll.eval(beta0, input, regularize=regularize, debug=False)
    (trial2, output2) = nll.eval(beta0, input, regularize=regularize, debug=False)
    assert trial1 == trial2
    assert numpy.all(output1 == output2)

    grad1 = nll.grad(beta0, input, regularize=regularize, debug=False)
    grad2 = nll.grad(beta0, input, regularize=regularize, debug=False)
    assert numpy.all(grad1 == grad2)

    def func(x: ndarray) -> float:
        return nll.eval(x, input, regularize=regularize, debug=False)[0]

    def grad(x: ndarray) -> ndarray:
        return nll.grad(x, input, regularize=regularize, debug=False)

    return func, grad