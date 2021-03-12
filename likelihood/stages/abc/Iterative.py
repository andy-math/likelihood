from __future__ import annotations

from abc import ABCMeta
from typing import Callable, Optional, Sequence, Tuple

import numpy
from likelihood.stages.abc.Stage import Stage
from numerical.typedefs import ndarray

_Iterative_gradinfo_t = Tuple[ndarray, ndarray, ndarray, ndarray]


class Iterative(Stage[_Iterative_gradinfo_t], metaclass=ABCMeta):
    output0: Optional[Callable[[ndarray], Tuple[ndarray, ndarray]]]
    evalf: Optional[Callable[[ndarray, ndarray, ndarray], ndarray]]
    gradf: Optional[
        Callable[
            [ndarray, ndarray, ndarray, ndarray, ndarray],
            Tuple[ndarray, ndarray, ndarray],
        ]
    ]

    def __init__(
        self,
        names: Sequence[str],
        input: Sequence[int],
        output: Sequence[int],
        output0: Callable[[ndarray], Tuple[ndarray, ndarray]],
        eval: Callable[[ndarray, ndarray, ndarray], ndarray],
        grad: Callable[
            [ndarray, ndarray, ndarray, ndarray, ndarray],
            Tuple[ndarray, ndarray, ndarray],
        ],
    ) -> None:
        super().__init__(names, input, output)
        self.output0 = output0
        self.evalf = eval
        self.gradf = grad

    def _eval(
        self, coeff: ndarray, inputs: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Iterative_gradinfo_t]]:
        assert self.output0 is not None
        assert self.evalf is not None
        output0, d0_dc = self.output0(coeff)
        nSample, nOutput = inputs.shape[0], output0.shape[0]
        outputs = numpy.ndarray((nSample, nOutput))
        outputs[0, :] = self.evalf(coeff, inputs[0, :], output0)
        for i in range(1, nSample):
            outputs[i, :] = self.evalf(coeff, inputs[i, :], outputs[i - 1, :])
        if not grad:
            return outputs, None
        return outputs, (output0, inputs, outputs, d0_dc)

    def _grad(
        self, coeff: ndarray, gradinfo: _Iterative_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        assert self.gradf is not None
        output0, inputs, outputs, d0_dc = gradinfo
        nSample, nInput = inputs.shape
        dL_di = numpy.ndarray((nSample, nInput))
        dL_dc = numpy.zeros(coeff.shape)
        for i in range(nSample - 1, 0, -1):
            _dL_dc, dL_di[i, :], _dL_do = self.gradf(
                coeff, inputs[i, :], outputs[i - 1, :], outputs[i, :], dL_do[i, :]
            )
            dL_dc += _dL_dc
            dL_do[i - 1, :] += _dL_do

        _dL_dc, dL_di[i, :], dL_d0 = self.gradf(
            coeff, inputs[0, :], output0, outputs[0, :], dL_do[0, :]
        )
        dL_dc += _dL_dc + dL_d0 @ d0_dc
        return dL_di, dL_dc
