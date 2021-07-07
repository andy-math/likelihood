from datetime import datetime
from typing import Callable, Tuple, TypeVar

import numpy
from likelihood.likelihood import negLikelihood
from likelihood.Variables import Variables
from overloads.typing import ndarray

T = TypeVar("T", int, datetime)


def nll2func(
    nll: negLikelihood, beta0: ndarray, data_in: Variables[T], *, regularize: bool
) -> Tuple[Callable[[ndarray], float], Callable[[ndarray], ndarray]]:
    (trial1, output1) = nll.eval(beta0, data_in, regularize=regularize, debug=True)
    (trial2, output2) = nll.eval(beta0, data_in, regularize=regularize, debug=True)
    assert trial1 == trial2
    assert numpy.all(output1 == output2)

    grad1 = nll.grad(beta0, data_in, regularize=regularize, debug=True)
    grad2 = nll.grad(beta0, data_in, regularize=regularize, debug=True)
    assert numpy.all(grad1 == grad2)

    (trial1, output1) = nll.eval(beta0, data_in, regularize=regularize, debug=False)
    (trial2, output2) = nll.eval(beta0, data_in, regularize=regularize, debug=False)
    assert trial1 == trial2
    assert numpy.all(output1 == output2)

    grad1 = nll.grad(beta0, data_in, regularize=regularize, debug=False)
    grad2 = nll.grad(beta0, data_in, regularize=regularize, debug=False)
    assert numpy.all(grad1 == grad2)

    def func(x: ndarray) -> float:
        return nll.eval(x, data_in, regularize=regularize, debug=False)[0]

    def grad(x: ndarray) -> ndarray:
        return nll.grad(x, data_in, regularize=regularize, debug=False)

    return func, grad
