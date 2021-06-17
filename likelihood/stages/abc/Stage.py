from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Generic, NamedTuple, Optional, Tuple, TypeVar

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
    names: Tuple[str, ...]
    data_in_index: Tuple[int, ...]
    data_out_index: Tuple[int, ...]

    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        data_in_index: Tuple[int, ...],
        data_out_index: Tuple[int, ...],
    ) -> None:
        assert len(set(coeff_names)) == len(coeff_names)
        assert len(set(data_in_index)) == len(data_in_index)
        assert len(set(data_out_index)) == len(data_out_index)
        super().__init__()
        self.names = coeff_names
        self.data_in_index = data_in_index
        self.data_out_index = data_out_index

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
        _input: ndarray = input[:, self.data_in_index]
        _output, gradinfo = self._eval(coeff, _input, grad=grad, debug=debug)
        assertNoInfNaN(_output)
        k = input.shape[0] - _output.shape[0]
        output = input[k:, :] if k else input
        output[:, self.data_out_index] = _output
        return output, gradinfo

    def grad(
        self, coeff: ndarray, gradinfo: _gradinfo_t, dL_do: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        _dL_do: ndarray = dL_do[:, self.data_out_index]
        dL_do[:, self.data_out_index] = 0.0
        _dL_di, dL_dc = self._grad(coeff, gradinfo, _dL_do, debug=debug)
        assertNoInfNaN(_dL_di)
        assertNoInfNaN(dL_dc)
        k = _dL_di.shape[0] - dL_do.shape[0]
        if k:
            dL_di = numpy.zeros((_dL_di.shape[0], dL_do.shape[1]))
            dL_di[k:, :] = dL_do
        else:
            dL_di = dL_do
        dL_di[:, self.data_in_index] += _dL_di
        return dL_di, dL_dc
