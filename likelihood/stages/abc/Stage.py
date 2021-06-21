from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Generic, List, NamedTuple, Optional, Tuple, TypeVar

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN, isunique


def _unique(t: Tuple[str, ...], *tuples: Tuple[str, ...]) -> Tuple[str, ...]:
    result: List[str] = list(t)
    for t in tuples:
        for s in t:
            if s not in result:
                result.append(s)
    return tuple(result)


class Constraints(NamedTuple):
    A: ndarray
    b: ndarray
    lb: ndarray
    ub: ndarray


_gradinfo_t = TypeVar("_gradinfo_t", contravariant=True)


class Stage(Generic[_gradinfo_t], metaclass=ABCMeta):
    coeff_names: Tuple[str, ...]
    coeff_index: Optional[ndarray] = None
    data_in_names: Tuple[str, ...]
    data_in_index: Tuple[int, ...]
    data_out_names: Tuple[str, ...]
    data_out_index: Tuple[int, ...]
    submodels: Tuple[Stage[Any], ...]

    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        data_in_names: Tuple[str, ...],
        data_out_names: Tuple[str, ...],
        data_in_index: Tuple[int, ...],
        data_out_index: Tuple[int, ...],
        submodels: Tuple[Stage[Any], ...],
    ) -> None:
        coeff_names = _unique(coeff_names, *(s.coeff_names for s in submodels))
        data_in_names = _unique(data_in_names, *(s.data_in_names for s in submodels))
        data_out_names = _unique(data_out_names, *(s.data_out_names for s in submodels))
        assert isunique(data_in_names)
        assert isunique(data_out_names)
        assert isunique(data_in_index)
        assert isunique(data_out_index)
        super().__init__()
        self.coeff_names = coeff_names
        self.data_in_names = data_in_names
        self.data_in_index = data_in_index
        self.data_out_names = data_out_names
        self.data_out_index = data_out_index
        self.submodels = submodels

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
    def get_constraints(
        self,
    ) -> Constraints:
        pass  # pragma: no cover

    def eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_gradinfo_t]]:
        """
        主要处理消除k-lag的问题
        对于长度缩小了的输出，从原input上截取k-lag项（也就是最后几项）
        然后将output贴进去
        另外，在此处对output检查是最经济的，只需要检查被输出的少数列，而不是更新后的完整sheet
        """
        _output, gradinfo = self._eval(
            coeff, input[:, self.data_in_index], grad=grad, debug=debug
        )
        assertNoInfNaN(_output)
        k = input.shape[0] - _output.shape[0]
        assert k >= 0
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
        assert k >= 0
        if k:
            dL_di = numpy.zeros((_dL_di.shape[0], dL_do.shape[1]))
            dL_di[k:, :] = dL_do
        else:
            dL_di = dL_do
        dL_di[:, self.data_in_index] += _dL_di
        return dL_di, dL_dc

    def register_coeff(
        self,
        likeli_names: Tuple[str, ...],
        data_in_names: Tuple[str, ...],
        data_out_names: Tuple[str, ...],
        register_constraints: Callable[[ndarray, Constraints], None],
    ) -> None:
        # 检查有无参数是未被声明的
        for x in self.coeff_names:
            if x not in likeli_names:
                assert False, f"模块{type(self).__name__}所使用的参数{x}未在似然函数中声明。"
        # 检查有无变量列名是未被声明的
        for x in self.data_in_names:
            if x not in data_in_names:
                assert False, f"模块{type(self).__name__}所使用的输入变量{x}未在似然函数中声明。"
        for x in self.data_out_names:
            if x not in data_out_names:
                assert False, f"模块{type(self).__name__}所使用的输出变量{x}未在似然函数中声明。"

        self.coeff_index = numpy.array(
            [likeli_names.index(x) for x in self.coeff_names], dtype=numpy.int64
        )
        data_in_index = numpy.array(
            [data_in_names.index(x) for x in self.data_in_names], dtype=numpy.int64
        )
        data_out_index = numpy.array(
            [data_out_names.index(x) for x in self.data_out_names], dtype=numpy.int64
        )
        assert numpy.all(self.data_in_index == data_in_index)
        assert numpy.all(self.data_out_index == data_out_index)
        register_constraints(self.coeff_index, self.get_constraints())

        def _register_constraints(
            coeff_index: ndarray, constraints: Constraints
        ) -> None:
            assert self.coeff_index is not None
            coeff_index = numpy.array(
                (self.coeff_index[i] for i in coeff_index), dtype=numpy.int64
            )
            register_constraints(coeff_index, constraints)

        for s in self.submodels:
            s.register_coeff(
                self.coeff_names,
                self.data_in_names,
                self.data_out_names,
                _register_constraints,
            )
