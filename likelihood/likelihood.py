from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN, isunique

from likelihood.Compose import Compose
from likelihood.stages.abc.Logpdf import Logpdf
from likelihood.stages.abc.Penalty import Penalty
from likelihood.stages.abc.Stage import Constraints, Stage


def _check_stages(stages: Tuple[Stage[Any], ...], nvars: int) -> None:
    for s in stages:
        assert not len(s.data_in_index) or max(s.data_in_index) < nvars
    assert isinstance(stages[-1], Logpdf)
    assert stages[-1].data_out_index[0] == 0


class negLikelihood:
    coeff_names: Tuple[str, ...]
    nInput: int
    stages: Tuple[Stage[Any], ...]
    penalty: Optional[Penalty[Any]]
    constraints: Constraints

    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        stages: Tuple[Stage[Any], ...],
        penalty: Optional[Penalty[Any]],
        *,
        nvars: int
    ) -> None:
        assert isunique(coeff_names)
        _check_stages(stages, nvars)
        self.coeff_names = coeff_names
        self.stages = stages
        self.penalty = penalty
        self.nInput = nvars
        self.constraints = Constraints(
            numpy.empty((0, len(coeff_names))),
            numpy.empty((0,)),
            numpy.full((len(coeff_names),), -numpy.inf),
            numpy.full((len(coeff_names),), numpy.inf),
        )
        for s in stages:
            s.register_coeff(coeff_names, self.register_constraints)
        if penalty is not None:
            penalty.register_coeff(coeff_names, self.register_constraints)

    def _get_stages(self, *, regularize: bool) -> Tuple[Stage[Any], ...]:
        if regularize:
            assert self.penalty is not None
            return self.stages + (self.penalty,)
        else:
            return self.stages

    def _eval(
        self: negLikelihood,
        coeff: ndarray,
        input: ndarray,
        *,
        grad: bool,
        regularize: bool,
        debug: bool
    ) -> Tuple[float, ndarray, Optional[Tuple[Any, ...]]]:

        assert coeff.shape == (len(self.coeff_names),)
        assert input.shape[1] == self.nInput

        assertNoInfNaN(coeff)
        assertNoInfNaN(input)

        output, gradinfo = Compose().eval(
            self._get_stages(regularize=regularize),
            coeff,
            input.copy(),
            grad=grad,
            debug=debug,
        )
        return -numpy.sum(output[:, 0]), output, gradinfo

    def eval(
        self, coeff: ndarray, input: ndarray, *, regularize: bool, debug: bool = False
    ) -> Tuple[float, ndarray]:
        fval, output, _ = self._eval(
            coeff, input, grad=False, regularize=regularize, debug=debug
        )
        return fval, output

    def grad(
        self, coeff: ndarray, input: ndarray, *, regularize: bool, debug: bool = False
    ) -> ndarray:
        _, o, gradinfo = self._eval(
            coeff, input, grad=True, regularize=regularize, debug=debug
        )

        dL_dL = numpy.zeros(o.shape)
        dL_dL[:, 0] = -1.0

        assert gradinfo is not None
        _, dL_dc = Compose().grad(
            self._get_stages(regularize=regularize), coeff, gradinfo, dL_dL, debug=debug
        )

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
        return self.constraints
