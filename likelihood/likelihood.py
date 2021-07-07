from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple, Type, TypeVar

import numpy
from overloads.typing import ndarray
from overloads.shortcuts import assertNoInfNaN, isunique

from likelihood.stages.abc.Logpdf import Logpdf
from likelihood.stages.abc.Penalty import Penalty
from likelihood.stages.abc.Stage import Constraints, Stage
from likelihood.stages.Mapping import Mapping
from likelihood.Variables import Variables

T = TypeVar("T", int, datetime)


def _isStage(s: Stage[Any], T: Type[Any]) -> bool:
    return isinstance(s, T) or (isinstance(s, Mapping) and isinstance(s.submodel, T))


def _eval_loop(
    stages: Tuple[Stage[Any], ...],
    coeff: ndarray,
    input: ndarray,
    *,
    grad: bool,
    debug: bool,
) -> Tuple[ndarray, Optional[Tuple[Any, ...]]]:
    output: ndarray = input
    gradinfo: List[Optional[Any]] = []
    for s in stages:
        assert s.coeff_index is not None
        output, g = s.eval(coeff[s.coeff_index], output, grad=grad, debug=debug)
        gradinfo.append(g)
    if not grad:
        return output, None
    return output, tuple(gradinfo)


def _grad_loop(
    stages: Tuple[Stage[Any], ...],
    coeff: ndarray,
    gradinfo: Tuple[Any, ...],
    dL_do: ndarray,
    *,
    debug: bool,
) -> Tuple[ndarray, ndarray]:
    dL_dc = numpy.zeros(coeff.shape)
    for s, g in zip(stages[::-1], gradinfo[::-1]):
        assert s.coeff_index is not None
        dL_do, _dL_dc = s.grad(coeff[s.coeff_index], g, dL_do, debug=debug)
        dL_dc[s.coeff_index] += _dL_dc
    return dL_do, dL_dc


def _check_stages(
    coeff_names: Tuple[str, ...], stages: Tuple[Stage[Any], ...], firstColName: str
) -> None:
    for name in coeff_names:
        if not any([name in s.coeff_names for s in stages]):
            assert False, f"likelihood中所声明的参数{name}似乎未被任何stage所引用"
    for s in stages:
        assert not _isStage(s, Penalty)
        assert isunique(s.coeff_names)
        assert isunique(s.data_in_names)
        assert isunique(s.data_out_names)
    assert _isStage(stages[-1], Logpdf)
    assert stages[-1].data_out_names[0] == firstColName


class negLikelihood:
    coeff_names: Tuple[str, ...]
    data_names: Tuple[str, ...]
    stages: Tuple[Stage[Any], ...]
    penalty: Optional[Penalty[Any]]
    constraints: Constraints

    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        data_names: Tuple[str, ...],
        stages: Tuple[Stage[Any], ...],
        penalty: Optional[Penalty[Any]],
    ) -> None:
        assert isunique(coeff_names)
        assert isunique(data_names)
        _check_stages(coeff_names, stages, data_names[0])
        self.coeff_names = coeff_names
        self.data_names = data_names
        self.stages = stages
        self.penalty = penalty
        self.constraints = Constraints(
            numpy.empty((0, len(coeff_names))),
            numpy.empty((0,)),
            numpy.full((len(coeff_names),), -numpy.inf),
            numpy.full((len(coeff_names),), numpy.inf),
        )
        for s in stages:
            s.register_coeff_and_data_names(
                coeff_names, data_names, data_names, self.register_constraints
            )
        if penalty is not None:
            penalty.register_coeff_and_data_names(
                coeff_names, data_names, data_names, self.register_constraints
            )

    def _get_stages(self, *, regularize: bool) -> Tuple[Stage[Any], ...]:
        if regularize:
            assert self.penalty is not None
            return self.stages + (self.penalty,)
        else:
            return self.stages

    def _eval(
        self: negLikelihood,
        coeff: ndarray,
        data_in: Variables[T],
        *,
        grad: bool,
        regularize: bool,
        debug: bool,
    ) -> Tuple[float, ndarray, Optional[Tuple[Any, ...]]]:

        assert coeff.shape == (
            len(self.coeff_names),
        ), "向negLikelihood所输入的参数向量的尺寸与预期的不同"
        assert (
            data_in.data_names == self.data_names
        ), "Variables中定义的变量与negLikelihood需要的变量似乎不同"

        assertNoInfNaN(coeff)
        assertNoInfNaN(data_in.sheet)

        output, gradinfo = _eval_loop(
            self._get_stages(regularize=regularize),
            coeff,
            data_in.sheet.copy(),
            grad=grad,
            debug=debug,
        )
        return -numpy.sum(output[:, 0]), output, gradinfo

    def eval(
        self,
        coeff: ndarray,
        data_in: Variables[T],
        *,
        regularize: bool,
        debug: bool = False,
    ) -> Tuple[float, ndarray]:
        fval, output, _ = self._eval(
            coeff, data_in, grad=False, regularize=regularize, debug=debug
        )
        return fval, output

    def grad(
        self,
        coeff: ndarray,
        data_in: Variables[T],
        *,
        regularize: bool,
        debug: bool = False,
    ) -> ndarray:
        _, o, gradinfo = self._eval(
            coeff, data_in, grad=True, regularize=regularize, debug=debug
        )

        dL_dL = numpy.zeros(o.shape)
        dL_dL[:, 0] = -1.0

        assert gradinfo is not None
        _, dL_dc = _grad_loop(
            self._get_stages(regularize=regularize), coeff, gradinfo, dL_dL, debug=debug
        )

        return dL_dc

    def register_constraints(
        self, coeff_index: ndarray, constraints: Constraints
    ) -> None:
        if not coeff_index.shape[0]:
            return
        self_A, self_b, self_lb, self_ub = self.constraints

        A = numpy.zeros((constraints.A.shape[0], len(self.coeff_names)))
        A[:, coeff_index] = constraints.A
        self_A = numpy.concatenate((self_A, A), axis=0)
        self_b = numpy.concatenate((self_b, constraints.b))
        self_lb[coeff_index] = numpy.maximum(self_lb[coeff_index], constraints.lb)
        self_ub[coeff_index] = numpy.minimum(self_ub[coeff_index], constraints.ub)

        self.constraints = Constraints(self_A, self_b, self_lb, self_ub)

    def get_constraints(self) -> Constraints:
        return self.constraints
