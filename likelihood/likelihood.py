from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

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


_gradinfo_t = TypeVar("_gradinfo_t")


class Stage(Generic[_gradinfo_t], metaclass=ABCMeta):
    @abstractproperty
    def names(self) -> List[str]:
        pass

    @abstractmethod
    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_gradinfo_t]]:
        pass

    @abstractmethod
    def _grad(
        self, coeff: ndarray, gradinfo: _gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        pass

    def eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_gradinfo_t]]:
        return self._eval(coeff, input, grad=grad)

    def grad(
        self, coeff: ndarray, gradinfo: _gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        return self._grad(coeff, gradinfo, dL_do)


_Compose_gradinfo_t = List[object]


class Compose(Stage[_Compose_gradinfo_t]):
    names: List[str]
    packing: List[int]
    stages: List[Stage[object]]

    def __init__(self, stages: List[Stage[object]]) -> None:
        self.names, self.packing = _make_names(*stages)
        self.stages = stages

    def _unpack(self, coeff: ndarray) -> List[ndarray]:
        return numpy.split(coeff, self.packing)

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Compose_gradinfo_t]]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        output: ndarray = input
        gradinfo: List[Optional[object]] = []
        for s, c in zip(stages, coeffs):
            output, g = s.eval(c, output, grad=grad)
            gradinfo.append(g)
        if not grad:
            return output, None
        return output, gradinfo

    def _grad(
        self, coeff: ndarray, gradinfo: _Compose_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        coeffs = self._unpack(coeff)
        stages = self.stages
        dL_dc: List[ndarray] = []
        for s, c, g in zip(stages[::-1], coeffs[::-1], gradinfo[::-1]):
            dL_do, _dL_dc = s.grad(c, g, dL_do)
            dL_dc.append(_dL_dc)
        return dL_do, numpy.concatenate(dL_dc[::-1])


_Elementwise_gradinfo_t = Tuple[ndarray, ndarray]


class Elementwise(Stage[_Elementwise_gradinfo_t]):
    names: List[str]
    evalf: Optional[Callable[[ndarray, ndarray], ndarray]]
    gradf: Optional[Callable[[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]]

    def __init__(
        self,
        names: List[str],
        func: Callable[[ndarray, ndarray], ndarray],
        grad: Callable[[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    ) -> None:
        self.names = names
        self.evalf = func
        self.gradf = grad

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Elementwise_gradinfo_t]]:
        assert self.evalf is not None
        output = self.evalf(coeff, input)
        if not grad:
            return output, None
        return output, (input, output)

    def _grad(
        self, coeff: ndarray, gradinfo: _Elementwise_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        assert self.gradf is not None
        input, output = gradinfo
        do_di, do_dc = self.gradf(coeff, input, output)
        dL_di = dL_do * do_di
        dL_dc = numpy.sum(dL_do * do_dc, axis=(0, 1))
        return dL_di, dL_dc


"""
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
"""
nSample, n = output.shape
do_di = numpy.tile(numpy.eye(n), [nSample, 1, 1])
do_dc = numpy.ndarray((nSample, nOutput, nCoeff))
for i in range(nSample - 1, -1, -1):
    pass
"""


class Likelihood:
    stages: Compose

    def __init__(self, stages: List[Stage[object]]) -> None:
        self.stages = Compose(stages)

    def eval(self, coeff: ndarray, input: ndarray) -> ndarray:
        o, _ = self.stages.eval(coeff, input, grad=False)
        return o

    def grad(self, coeff: ndarray, input: ndarray) -> ndarray:
        _, gradinfo = self.stages.eval(coeff, input, grad=True)
        assert gradinfo is not None
        _, dL_dc = self.stages.grad(coeff, gradinfo, numpy.array([1.0]))
        return dL_dc
