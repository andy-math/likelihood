from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN

from likelihood.Compose import Compose
from likelihood.stages.abc.Logpdf import Logpdf
from likelihood.stages.abc.Penalty import Penalty
from likelihood.stages.abc.Stage import Constraints, Stage


def _check_stages(stages: List[Stage[Any]], nvars: int) -> None:
    for s in stages:
        assert not len(s.data_in_index) or max(s.data_in_index) < nvars
    assert isinstance(stages[-1], Logpdf)
    assert stages[-1].data_out_index[0] == 0


class negLikelihood:
    coeff_names: Tuple[str, ...]
    nInput: int
    stages: Compose
    penalty: Optional[Penalty[Any]]
    constraints: Constraints

    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        stages: List[Stage[Any]],
        penalty: Optional[Penalty[Any]],
        *,
        nvars: int
    ) -> None:
        _check_stages(stages, nvars)
        self.coeff_names = coeff_names
        self.stages = Compose(stages, nvars)
        self.penalty = penalty
        self.nInput = nvars
        if penalty is not None:
            penalty.make_index(coeff_names)
        self.constraints = Constraints(
            numpy.empty((0, len(coeff_names))),
            numpy.empty((0,)),
            numpy.full((len(coeff_names),), -numpy.inf),
            numpy.full((len(coeff_names),), numpy.inf),
        )
        for s in stages:
            s.register_coeff(coeff_names, self.register_constraints)

    def _eval(
        self: negLikelihood,
        coeff: ndarray,
        input: ndarray,
        *,
        grad: bool,
        regularize: bool,
        debug: bool
    ) -> Tuple[float, ndarray, Optional[Any], Optional[Any]]:

        assert coeff.shape == (len(self.coeff_names),)
        assert input.shape[1] == self.nInput

        assertNoInfNaN(coeff)
        assertNoInfNaN(input)

        output, gradinfo = self.stages.eval(coeff, input.copy(), grad=grad, debug=debug)
        _gradinfo = None
        if regularize:
            assert self.penalty is not None
            assert self.penalty.index is not None
            index = self.penalty.index
            output, _gradinfo = self.penalty.eval(
                coeff[index], output, grad=grad, debug=debug
            )
        return -numpy.sum(output[:, 0]), output, gradinfo, _gradinfo

    def eval(
        self, coeff: ndarray, input: ndarray, *, regularize: bool, debug: bool = False
    ) -> Tuple[float, ndarray]:
        fval, output, _, _ = self._eval(
            coeff, input, grad=False, regularize=regularize, debug=debug
        )
        return fval, output

    def grad(
        self, coeff: ndarray, input: ndarray, *, regularize: bool, debug: bool = False
    ) -> ndarray:
        _, o, gradinfo, _gradinfo = self._eval(
            coeff, input, grad=True, regularize=regularize, debug=debug
        )

        dL_dL = numpy.zeros(o.shape)
        dL_dL[:, 0] = -1.0

        if regularize:
            assert self.penalty is not None
            assert _gradinfo is not None
            assert self.penalty.index is not None
            index = self.penalty.index
            dL_dL, _dL_dc = self.penalty.grad(
                coeff[index], _gradinfo, dL_dL, debug=debug
            )

        assert gradinfo is not None
        _, dL_dc = self.stages.grad(coeff, gradinfo, dL_dL, debug=debug)

        if regularize:
            dL_dc[index] += _dL_dc

        return dL_dc

    def register_constraints(
        self, coeff_index: ndarray, constraints: Constraints
    ) -> None:
        if not coeff_index.shape[0]:
            return
        self_A, self_b, self_lb, self_ub = self.constraints

        self_A = numpy.concatenate(
            (self_A, numpy.zeros((constraints.A.shape[0], len(self.coeff_names)))),
            axis=0,
        )
        self_A[-constraints.A.shape[0] :, coeff_index] = constraints.A
        self_b = numpy.concatenate((self_b, constraints.b))
        self_lb[coeff_index] = numpy.maximum(self_lb[coeff_index], constraints.lb)
        self_ub[coeff_index] = numpy.minimum(self_ub[coeff_index], constraints.ub)

        self.constraints = Constraints(self_A, self_b, self_lb, self_ub)

    def get_constraints(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        const = self.stages.get_constraints()
        assert numpy.all(const.A == self.constraints.A)
        assert numpy.all(const.b == self.constraints.b)
        assert numpy.all(const.lb == self.constraints.lb)
        assert numpy.all(const.ub == self.constraints.ub)
        return const
