from __future__ import annotations

from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Logpdf import Logpdf
from numerical.typedefs import ndarray

_LogNormpdf_gradinfo_t = ndarray


class LogNormpdf(Logpdf[_LogNormpdf_gradinfo_t]):
    def __init__(self, variance_name: str, input: Tuple[int, int], output: int) -> None:
        super().__init__([variance_name], input, (output,))

    def _eval(
        self, var: ndarray, mu_x: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_LogNormpdf_gradinfo_t]]:
        """
        x ~ N(0, Var)
        p(x) = 1/sqrt(Var*2pi) * exp{ -(x*x)/(2Var) }
        log p(x) = -1/2{ log(Var) + log(2pi) } - (x*x)/(2Var)
                 = (-1/2) { log(Var) + log(2pi) + (x*x)/Var }
        """
        x: ndarray = mu_x[:, [1]] - mu_x[:, [0]]  # type: ignore
        constant = numpy.log(var) + numpy.log(2.0) + numpy.log(numpy.pi)
        logP = (-1.0 / 2.0) * (constant + (x * x) / var)
        if not grad:
            return logP, None
        return logP, x

    def _grad(
        self, var: ndarray, x: _LogNormpdf_gradinfo_t, dL_dlogP: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """
        d/dx{log p(x)} = (-1/2) { 2x/Var } = -x/Var
        d/dVar{log p(x)} = (-1/2) {1/Var - (x*x)/(Var*Var)}
                         = (1/2) {(x/Var) * (x/Var) - 1/Var}
        """
        z = x / var
        dL_di = dL_dlogP * -z
        dL_dlogP.shape = (dL_dlogP.shape[0],)
        dL_dc = dL_dlogP @ ((1.0 / 2.0) * (z * z - 1.0 / var))
        return dL_di, dL_dc
