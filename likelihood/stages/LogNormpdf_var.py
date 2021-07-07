from __future__ import annotations

import math
from typing import Optional, Tuple, cast

import numpy
from likelihood.stages.abc.Logpdf import Logpdf
from likelihood.stages.abc.Stage import Constraints
from overloads.typing import ndarray

_LogNormpdfVar_gradinfo_t = Tuple[ndarray, ndarray]


class LogNormpdf_var(Logpdf[_LogNormpdfVar_gradinfo_t]):
    def __init__(
        self, data_in_names: Tuple[str, str], data_out_names: Tuple[str, str]
    ) -> None:
        super().__init__((), data_in_names, data_out_names, ())

    def _eval(
        self, _: ndarray, x_var: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_LogNormpdfVar_gradinfo_t]]:
        """
        x ~ N(0, Var)
        p(x) = 1/sqrt(Var*2pi) * exp{ -(x*x)/(2Var) }
        log p(x) = -1/2{ log(Var) + log(2pi) } - (x*x)/(2Var)
                 = (-1/2) { log(Var) + log(2pi) + (x*x)/Var }
        """
        x: ndarray = x_var[:, [0]]
        var: ndarray = x_var[:, [1]]
        constant = math.log(2.0) + math.log(math.pi)
        logP = (-1.0 / 2.0) * (numpy.log(var) + (x * x) / var + constant)
        output: ndarray = numpy.concatenate((logP, var), axis=1)  # type: ignore
        if not grad:
            return output, None
        return output, (x, var)

    def _grad(
        self,
        var: ndarray,
        x_var: _LogNormpdfVar_gradinfo_t,
        dL_dlogP_dvar: ndarray,
        *,
        debug: bool
    ) -> Tuple[ndarray, ndarray]:
        """
        d/dx{log p(x)} = (-1/2) { 2x/Var } = -x/Var
        d/dVar{log p(x)} = (-1/2) {1/Var - (x*x)/(Var*Var)}
                         = (1/2) {(x/Var) * (x/Var) - 1/Var}
        """
        x, var = x_var
        z: ndarray = x / var
        dL_dlogP: ndarray = dL_dlogP_dvar[:, [0]]
        dL_dvar: ndarray = dL_dlogP_dvar[:, [1]]
        dL_dx: ndarray = dL_dlogP * -z
        dL_dvar = dL_dvar + dL_dlogP * (
            (1.0 / 2.0) * (cast(ndarray, z * z) - 1.0 / var)
        )
        return (
            numpy.concatenate((dL_dx, dL_dvar), axis=1),  # type: ignore
            numpy.ndarray((0,)),
        )

    def get_constraints(self) -> Constraints:
        A = numpy.empty((0, 0))
        b = numpy.empty((0,))
        lb = numpy.empty((0,))
        ub = numpy.empty((0,))
        return Constraints(A, b, lb, ub)
