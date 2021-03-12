from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy
from numerical.typedefs import ndarray

from likelihood.stages.abc.Logpdf import Logpdf
from likelihood.stages.abc.Stage import Stage
from likelihood.stages.Compose import Compose

_Iterative_gradinfo_t = Tuple[ndarray, ndarray]


class Iterative(Stage[_Iterative_gradinfo_t]):
    names: List[str]
    output0: ndarray
    evalf: Optional[Callable[[ndarray, ndarray, ndarray], ndarray]]
    gradf: Optional[
        Callable[[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray]]
    ]

    def _eval(
        self, coeff: ndarray, inputs: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Iterative_gradinfo_t]]:
        assert self.evalf is not None
        output0 = self.output0
        nSample, nOutput = inputs.shape[0], output0.shape[0]
        outputs = numpy.ndarray((nSample, nOutput))
        for i in range(nSample):
            output0 = self.evalf(coeff, inputs[i, :], output0)
            outputs[i, :] = output0
        if not grad:
            return outputs, None
        return outputs, (inputs, outputs)

    def _grad(
        self, coeff: ndarray, gradinfo: _Iterative_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        assert self.gradf is not None
        inputs, outputs = gradinfo
        nSample, nInput = inputs.shape
        assert outputs.shape[0] == nSample and dL_do.shape[0] == nSample
        dL_di = numpy.ndarray((nSample, nInput))
        dL_dc = numpy.zeros(coeff.shape)
        for i in range(nSample - 1, 0, -1):
            do_di, do_do, do_dc = self.gradf(
                coeff, inputs[i, :], outputs[i - 1, :], outputs[i, :]
            )
            dL_di[i, :] = dL_do[i, :] @ do_di
            dL_dc += dL_do[i, :] @ do_dc
            dL_do[i - 1, :] += dL_do[i, :] @ do_do

        do_di, do_do, do_dc = self.gradf(
            coeff, inputs[0, :], self.output0, outputs[0, :]
        )
        dL_di[0, :] = dL_do[0, :] @ do_di
        dL_dc += dL_do[0, :] @ do_dc
        return dL_di, dL_dc


class negLikelihood:
    nCoeff: int
    nInput: int
    stages: Compose

    def __init__(self, stages: List[Stage[Any]], nVars: int) -> None:
        assert isinstance(stages[-1], Logpdf)
        assert len(stages[-1]._output_idx) == 1
        assert stages[-1]._output_idx[0] == 0
        self.stages = Compose(stages, list(range(nVars)), list(range(nVars)))
        self.nCoeff = self.stages.len_coeff
        self.nInput = nVars

    def eval(self, coeff: ndarray, input: ndarray) -> float:
        assert coeff.shape == (self.nCoeff,)
        o, _ = self.stages.eval(coeff, input.copy(), grad=False)
        return -numpy.sum(o[:, 0])

    def grad(self, coeff: ndarray, input: ndarray) -> ndarray:
        assert coeff.shape == (self.nCoeff,)
        o, gradinfo = self.stages.eval(coeff, input.copy(), grad=True)
        assert gradinfo is not None
        dL_dL = numpy.zeros(o.shape)
        dL_dL[:, 0] = -1.0
        _, dL_dc = self.stages.grad(coeff, gradinfo, dL_dL)
        return dL_dc
