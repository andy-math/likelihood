from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray


def _make_names(*stages: Stage) -> Tuple[List[str], List[int]]:
    names: List[str] = []
    packing: List[int] = []
    for s in stages:
        names.extend(s.names)
        packing.append(len(s.names))
    assert len(set(names)) == len(names)
    return names, numpy.cumsum(packing)  # type: ignore


class Stage(ABC):
    @abstractproperty
    def names(self) -> List[str]:
        pass

    def eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[Tuple[ndarray, ndarray]]]:
        return self._eval(coeff, input, grad=grad)

    @abstractmethod
    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[Tuple[ndarray, ndarray]]]:
        pass


class Compose(Stage):
    names: List[str]
    packing: List[int]
    stages: List[Stage]

    def __init__(self, stages: List[Stage]) -> None:
        self.names, self.packing = _make_names(*stages)
        self.stages = stages

    def _unpack(self, coeff: ndarray) -> List[ndarray]:
        return numpy.split(coeff, self.packing)

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[Tuple[ndarray, ndarray]]]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        output: ndarray = input
        grads: List[Optional[Tuple[ndarray, ndarray]]] = []
        for s, c in zip(stages, coeffs):
            output, g = s.eval(c, output, grad=grad)
            grads.append(g)
        if not grad:
            return output, None
        nSample, n = output.shape
        do_di = numpy.tile(numpy.eye(n), [nSample, 1, 1])
        do_dc: List[ndarray] = []
        for g in grads[::-1]:
            assert g is not None
            _do_di, _do_dc = g
            do_dc.append(do_di @ _do_dc)
            do_di = do_di @ _do_di
        return output, (do_di, numpy.concatenate(do_dc))


class Elementwise(Stage):
    names: List[str]
    func: Optional[Callable[[ndarray, ndarray], ndarray]]
    grad: Optional[Callable[[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]]

    def __init__(
        self,
        names: List[str],
        func: Callable[[ndarray, ndarray], ndarray],
        grad: Callable[[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    ) -> None:
        self.names = names
        self.func = func
        self.grad = grad

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[Tuple[ndarray, ndarray]]]:
        assert self.func is not None
        assert self.grad is not None
        output = self.func(coeff, input)
        if not grad:
            return output, None
        return output, self.grad(coeff, input, output)


class Iterative(Stage):
    names: List[str]
    output0: ndarray
    func: Optional[
        Callable[[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]]
    ]

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[Tuple[ndarray, ndarray]]]:
        assert self.func is not None
        (nCoeff,) = coeff.shape
        nSample, nInput = input.shape
        (nOutput,) = self.output0.shape
        output0 = self.output0
        output = numpy.ndarray((nSample, nOutput))
        _do_di = numpy.ndarray((nSample, nOutput, nInput))
        _do_do = numpy.ndarray((nSample, nOutput, nOutput))
        _do_dc = numpy.ndarray((nSample, nOutput, nCoeff))
        for i in range(nSample):
            output0, _do_di[i, :, :], _do_do[i, :, :], _do_dc[i, :, :] = self.func(
                coeff, input[i, :], output0
            )
            output[i, :] = output0
        if not grad:
            return output, None
        """
        nSample, n = output.shape
        do_di = numpy.tile(numpy.eye(n), [nSample, 1, 1])
        do_dc = numpy.ndarray((nSample, nOutput, nCoeff))
        for i in range(nSample - 1, -1, -1):
            pass
        """


class Likelihood:
    stages: Compose

    def __init__(self, stages: List[Stage]) -> None:
        self.stages = Compose(stages)

    def eval(self, coeff: ndarray, input: ndarray) -> ndarray:
        o, _ = self.stages.eval(coeff, input, grad=False)
        return o

    def grad(self, coeff: ndarray, input: ndarray) -> ndarray:
        _, g = self.stages.eval(coeff, input, grad=True)
        assert g is not None
        _, do_dc = g
        return do_dc
