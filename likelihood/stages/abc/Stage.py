from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Generic, List, NamedTuple, Optional, Sequence, Tuple, TypeVar

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN


class Constraints(NamedTuple):
    A: ndarray
    b: ndarray
    lb: ndarray
    ub: ndarray


_gradinfo_t = TypeVar("_gradinfo_t", contravariant=True)


class Stage(Generic[_gradinfo_t], metaclass=ABCMeta):
    names: List[str]
    _input_idx: Sequence[int]
    _output_idx: Sequence[int]

    def __init__(
        self, names: Sequence[str], input: Sequence[int], output: Sequence[int]
    ) -> None:
        assert len(set(names)) == len(names)
        assert len(set(input)) == len(input)
        assert len(set(output)) == len(output)
        super().__init__()
        self.names = list(names)
        self._input_idx = input
        self._output_idx = output

    @abstractmethod
    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_gradinfo_t]]:
        pass  # pragma: no cover

    @abstractmethod
    def _grad(
        self, coeff: ndarray, gradinfo: _gradinfo_t, dL_do: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        pass  # pragma: no cover

    @abstractmethod
    def get_constraint(
        self,
    ) -> Constraints:
        pass  # pragma: no cover

    def eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_gradinfo_t]]:
        _input: ndarray = input[:, self._input_idx]
        _output, gradinfo = self._eval(coeff, _input, grad=grad, debug=debug)
        assertNoInfNaN(_output)
        k = input.shape[0] - _output.shape[0]
        output = input[k:, :] if k else input
        output[:, self._output_idx] = _output
        return output, gradinfo

    def grad(
        self, coeff: ndarray, gradinfo: _gradinfo_t, dL_do: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        _dL_do: ndarray = dL_do[:, self._output_idx]
        dL_do[:, self._output_idx] = 0.0
        _dL_di, dL_dc = self._grad(coeff, gradinfo, _dL_do, debug=debug)
        assertNoInfNaN(_dL_di)
        assertNoInfNaN(dL_dc)
        k = _dL_di.shape[0] - dL_do.shape[0]
        if k:
            dL_di = numpy.zeros((_dL_di.shape[0], dL_do.shape[1]))
            dL_di[k:, :] = dL_do
        else:
            dL_di = dL_do
        dL_di[:, self._input_idx] += _dL_di
        return dL_di, dL_dc
