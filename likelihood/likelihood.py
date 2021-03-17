from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN

from likelihood.stages.abc.Logpdf import Logpdf
from likelihood.stages.abc.Penalty import Penalty
from likelihood.stages.abc.Stage import Stage
from likelihood.stages.Compose import Compose


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

    def eval(self, coeff: ndarray, input: ndarray, *, regularize: bool) -> Tuple[float, ndarray]:
        assert coeff.shape == (self.nCoeff,)
        assertNoInfNaN(input)
        o, _ = self.stages.eval(coeff, input.copy(), grad=False)
        if regularize:
            assert self.penalty is not None
            o, _ = self.penalty.eval(coeff, o, grad=False)
        return -numpy.sum(o[:, 0]), o

    def grad(self, coeff: ndarray, input: ndarray, *, regularize: bool) -> ndarray:
        assert coeff.shape == (self.nCoeff,)
        assertNoInfNaN(input)
        o, gradinfo = self.stages.eval(coeff, input.copy(), grad=True)
        assert gradinfo is not None

        if regularize:
            assert self.penalty is not None
            o, _gradinfo = self.penalty.eval(coeff, o, grad=True)
            assert _gradinfo is not None

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
