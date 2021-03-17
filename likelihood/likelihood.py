from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN

from likelihood.stages.abc.Logpdf import Logpdf
from likelihood.stages.abc.Penalty import Penalty
from likelihood.stages.abc.Stage import Stage
from likelihood.stages.Compose import Compose


def _eval(
    self: negLikelihood, coeff: ndarray, input: ndarray, *, grad: bool, regularize: bool
) -> Tuple[float, ndarray, Optional[Any], Optional[Any]]:
    assert coeff.shape == (self.nCoeff,)
    assertNoInfNaN(input)
    output, gradinfo = self.stages.eval(coeff, input.copy(), grad=grad)
    _gradinfo = None
    if regularize:
        assert self.penalty is not None
        output, _gradinfo = self.penalty.eval(coeff, output, grad=grad)
    return -numpy.sum(output[:, 0]), output, gradinfo, _gradinfo


class negLikelihood:
    nCoeff: int
    nInput: int
    stages: Compose
    penalty: Optional[Penalty[Any]]

    def __init__(
        self, stages: List[Stage[Any]], penalty: Optional[Penalty[Any]], *, nvars: int
    ) -> None:
        assert isinstance(stages[-1], Logpdf)
        assert stages[-1]._output_idx[0] == 0
        self.stages = Compose(stages, list(range(nvars)), list(range(nvars)))
        self.penalty = penalty
        self.nCoeff = self.stages.len_coeff
        self.nInput = nvars

    def eval(
        self, coeff: ndarray, input: ndarray, *, regularize: bool
    ) -> Tuple[float, ndarray]:
        fval, output, _, _ = _eval(
            self, coeff, input, grad=False, regularize=regularize
        )
        return fval, output

    def grad(self, coeff: ndarray, input: ndarray, *, regularize: bool) -> ndarray:
        _, o, gradinfo, _gradinfo = _eval(
            self, coeff, input, grad=True, regularize=regularize
        )
        assert gradinfo is not None and _gradinfo is not None

        dL_dL = numpy.zeros(o.shape)
        dL_dL[:, 0] = -1.0
        _dL_dc = numpy.zeros(coeff.shape)

        if regularize:
            assert self.penalty is not None
            dL_dL, _dL_dc = self.penalty.grad(coeff, _gradinfo, dL_dL)

        _, dL_dc = self.stages.grad(coeff, gradinfo, dL_dL)
        dL_dc += _dL_dc
        return dL_dc

    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return self.stages.get_constraint()
