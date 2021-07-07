from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import numpy
from likelihood.stages.abc.Stage import Constraints, Stage
from overloads.typing import ndarray
from overloads.shortcuts import isunique

T = TypeVar("T")


class Mapping(Stage[T]):
    submodel: Stage[T]
    expand_index: ndarray

    def __init__(
        self, _mapping: Dict[str, Tuple[str, ...]], submodel: Stage[T]
    ) -> None:
        expanded_names = tuple(y for x in _mapping.values() for y in x)
        assert isunique(expanded_names)

        for name in submodel.coeff_names:
            if name not in expanded_names:
                assert False, f"Mapping中的子模型所使用的参数{name}未在Mapping层声明"

        for name in expanded_names:
            if name not in submodel.coeff_names:
                assert False, f"Mapping中所声明的参数{name}未被子模型引用"

        mapping = tuple((k, v) for k, v in _mapping.items())
        coeff_names = tuple(k for k, _ in mapping)

        super().__init__(
            coeff_names, submodel.data_in_names, submodel.data_out_names, ()
        )

        self.submodel = submodel

        expand_index: List[int] = []
        for name in submodel.coeff_names:
            for i, (_, v) in enumerate(mapping):
                if name in v:
                    expand_index.append(i)
        assert len(expand_index) == len(submodel.coeff_names)
        self.expand_index = numpy.array(expand_index, dtype=numpy.int64)

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[T]]:
        return self.submodel._eval(
            coeff[self.expand_index], input, grad=grad, debug=debug
        )

    def _grad(
        self, coeff: ndarray, gradinfo: T, dL_do: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        dL_di, _dL_dc = self.submodel._grad(
            coeff[self.expand_index], gradinfo, dL_do, debug=debug
        )
        dL_dc = numpy.zeros((len(self.coeff_names),))
        for i, v in zip(self.expand_index, _dL_dc):
            dL_dc[i] += v
        return dL_di, dL_dc

    def get_constraints(self) -> Constraints:
        assert False

    def register_coeff_and_data_names(
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
        self.data_in_index = numpy.array(
            [data_in_names.index(x) for x in self.data_in_names], dtype=numpy.int64
        )
        self.data_out_index = numpy.array(
            [data_out_names.index(x) for x in self.data_out_names], dtype=numpy.int64
        )

        def _register_constraints(index: ndarray, constraints: Constraints) -> None:
            assert self.coeff_index is not None
            index = self.expand_index[index]
            A = numpy.zeros((constraints.A.shape[0], len(self.coeff_names)))
            lb = numpy.full((len(self.coeff_names),), -numpy.inf)
            ub = numpy.full((len(self.coeff_names),), numpy.inf)

            for i, idx in enumerate(index):
                A[:, idx] = A[:, idx] + constraints.A[:, i]
                lb[idx] = max(lb[idx], constraints.lb[i])
                ub[idx] = min(ub[idx], constraints.ub[i])

            register_constraints(
                self.coeff_index, Constraints(A, constraints.b, lb, ub)
            )

        self.submodel.register_coeff_and_data_names(
            self.submodel.coeff_names,
            self.data_in_names,
            self.data_out_names,
            _register_constraints,
        )
