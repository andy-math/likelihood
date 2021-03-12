from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN


def _make_names(*stages: Stage[object]) -> Tuple[List[str], List[int]]:
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

    @abstractproperty
    def _input_idx(self) -> Sequence[int]:
        pass

    @abstractproperty
    def _output_idx(self) -> Sequence[int]:
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
        _input: ndarray = input[:, self._input_idx]  # type: ignore
        _output, gradinfo = self._eval(coeff, _input, grad=grad)
        assertNoInfNaN(_output)
        output = input
        output[:, self._output_idx] = _output
        return output, gradinfo

    def grad(
        self, coeff: ndarray, gradinfo: _gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        _dL_do: ndarray = dL_do[:, self._output_idx]  # type: ignore
        dL_do[:, self._output_idx] = 0.0
        _dL_di, dL_dc = self._grad(coeff, gradinfo, _dL_do)
        assertNoInfNaN(_dL_di)
        assertNoInfNaN(dL_dc)
        dL_di = dL_do
        dL_di[:, self._input_idx] = _dL_di
        return dL_di, dL_dc


_Compose_gradinfo_t = List[object]


class Compose(Stage[_Compose_gradinfo_t]):
    names: List[str]
    packing: List[int]
    _input_idx: Sequence[int]
    _output_idx: Sequence[int]
    stages: List[Stage[object]]

    def __init__(
        self, stages: List[Stage[object]], input: Sequence[int], output: Sequence[int]
    ) -> None:
        self.names, self.packing = _make_names(*stages)
        self._input_idx = input
        self._output_idx = output
        self.stages = stages
        assert len(input) == len(output)
        for s in stages:
            assert max(s._input_idx) < len(input)
            assert max(s._output_idx) < len(output)

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
    _input_idx: Sequence[int]
    _output_idx: Sequence[int]
    evalf: Optional[Callable[[ndarray, ndarray], ndarray]]
    gradf: Optional[Callable[[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]]

    def __init__(
        self,
        names: List[str],
        input: Sequence[int],
        output: Sequence[int],
        func: Callable[[ndarray, ndarray], ndarray],
        grad: Callable[[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    ) -> None:
        self.names = names
        self._input_idx = input
        self._output_idx = output
        self.evalf = func
        self.gradf = grad
        assert len(input) == len(output)

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
_Iterative_gradinfo_t = Tuple[ndarray, ndarray]


class Iterative(Stage[_Iterative_gradinfo_t]):
    names: List[str]
    output0: ndarray
    evalf: Optional[Callable[[ndarray, ndarray, ndarray], ndarray]]
    gradf: Optional[
        Callable[[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray]]
    ]

    def _eval(
        self, coeff: ndarray, inputs: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Iterative_gradinfo_t]]:
        assert self.evalf is not None
        output0 = self.output0
        nSample, nOutput = inputs.shape[0], output0.shape[0]
        outputs = numpy.ndarray((nSample, nOutput))
        for i in range(nSample):
            output0 = self.evalf(coeff, inputs[i, :], output0)
            outputs[i, :] = output0
        if not grad:
            return outputs, None
        return outputs, (inputs, outputs)

    def _grad(
        self, coeff: ndarray, gradinfo: _Iterative_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        assert self.gradf is not None
        inputs, outputs = gradinfo
        nSample, nInput = inputs.shape
        assert outputs.shape[0] == nSample and dL_do.shape[0] == nSample
        dL_di = numpy.ndarray((nSample, nInput))
        dL_dc = numpy.zeros(coeff.shape)
        for i in range(nSample - 1, 0, -1):
            do_di, do_do, do_dc = self.gradf(
                coeff, inputs[i, :], outputs[i - 1, :], outputs[i, :]
            )
            dL_di[i, :] = dL_do[i, :] @ do_di
            dL_dc += dL_do[i, :] @ do_dc
            dL_do[i - 1, :] += dL_do[i, :] @ do_do

        do_di, do_do, do_dc = self.gradf(
            coeff, inputs[0, :], self.output0, outputs[0, :]
        )
        dL_di[0, :] = dL_do[0, :] @ do_di
        dL_dc += dL_do[0, :] @ do_dc
        return dL_di, dL_dc
"""

"""
_Convolution_gradinfo_t = type(None)


class Convolution(Stage[_Convolution_gradinfo_t]):
    pass
"""

_Sum_gradinfo_t = int


class Sum(Stage[_Sum_gradinfo_t]):
    name: List[str] = []

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, int]:
        return numpy.sum(input, axis=0), input.shape[0]

    def _grad(
        self, coeff: ndarray, len: _Sum_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        return numpy.full((len, 1), dL_do), numpy.array([], dtype=numpy.float64)


_Linear_gradinfo_t = ndarray


class Linear(Stage[_Linear_gradinfo_t]):
    names: List[str]

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Linear_gradinfo_t]]:
        if not grad:
            return input @ coeff, None
        return input @ coeff, input

    def _grad(
        self, coeff: ndarray, input: _Linear_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        return dL_do * coeff, dL_do.flatten() @ input


class LogNormpdf(Elementwise):
    def __init__(
        self, names: List[str], input: Sequence[int], output: Sequence[int]
    ) -> None:
        def func(var: ndarray, x: ndarray) -> ndarray:
            """
            x ~ N(0, Var)
            p(x) = 1/sqrt(Var*2pi) * exp{ -(x*x)/(2Var) }
            log p(x) = -1/2{ log(Var) + log(2pi) } - (x*x)/(2Var)
                     = (-1/2) { log(Var) + log(2pi) + (x*x)/Var }
            """
            constant = numpy.log(var) + numpy.log(2.0) + numpy.log(numpy.pi)
            return (-1.0 / 2.0) * (constant + (x * x) / var)

        def grad(var: ndarray, x: ndarray, logP: ndarray) -> Tuple[ndarray, ndarray]:
            """
            d/dx{log p(x)} = (-1/2) { 2x/Var } = -x/Var
            d/dVar{log p(x)} = (-1/2) {1/Var - (x*x)/(Var*Var)}
                             = (1/2) {(x/Var) * (x/Var) - 1/Var}
            """
            z = x / var
            do_di = -z
            do_dc = (1.0 / 2.0) * (z * z - 1.0 / var)
            return do_di, do_dc

        assert len(names) == 1
        super().__init__(names, input, output, func, grad)


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
