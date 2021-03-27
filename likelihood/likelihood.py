from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN

from likelihood.stages.abc.Logpdf import Logpdf
from likelihood.stages.abc.Penalty import Penalty
from likelihood.stages.abc.Stage import Stage
from likelihood.stages.Compose import Compose


def _check_stages(stages: List[Stage[Any]], nvars: int) -> None:
    for s in stages:
        assert max(s._input_idx) < nvars
    assert isinstance(stages[-1], Logpdf)
    assert stages[-1]._output_idx[0] == 0


class negLikelihood:
    nCoeff: int
    nInput: int
    stages: Compose
    penalty: Optional[Penalty[Any]]

    def __init__(
        self, stages: List[Stage[Any]], penalty: Optional[Penalty[Any]], *, nvars: int
    ) -> None:
        _check_stages(stages, nvars)
        self.stages = Compose(stages, list(range(nvars)), list(range(nvars)))
        self.penalty = penalty
        self.nCoeff = self.stages.len_coeff
        self.nInput = nvars
        if penalty is not None:
            penalty.make_index(self.stages.names)

    def _eval(
        self: negLikelihood,
        coeff: ndarray,
        input: ndarray,
        *,
        grad: bool,
        regularize: bool,
        debug: bool
    ) -> Tuple[float, ndarray, Optional[Any], Optional[Any]]:

        assert coeff.shape == (self.nCoeff,)
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

    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return self.stages.get_constraint()
