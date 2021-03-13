from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar

from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN

_gradinfo_t = TypeVar("_gradinfo_t", contravariant=True)


class Stage(Generic[_gradinfo_t], metaclass=ABCMeta):
    names: List[str]
    _input_idx: Sequence[int]
    _output_idx: Sequence[int]

    def __init__(
        self, names: Sequence[str], input: Sequence[int], output: Sequence[int]
    ) -> None:
        assert len(set(names)) == len(names)
        super().__init__()
        self.names = list(names)
        self._input_idx = input
        self._output_idx = output

    @abstractmethod
    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_gradinfo_t]]:
        pass  # pragma: no cover

    @abstractmethod
    def _grad(
        self, coeff: ndarray, gradinfo: _gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        pass  # pragma: no cover

    @abstractmethod
    def get_constraint(
        self,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        pass  # pragma: no cover

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
