from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Penalty import Penalty
from overloads.typedefs import ndarray

_Lasso_gradinfo_t = Tuple[ndarray]


class Lasso(Penalty[_Lasso_gradinfo_t]):
    Lambda: float

    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        Lambda: float,
        data_in_names: Tuple[str, str],
        data_out_name: str,
    ) -> None:
        super().__init__(coeff_names, data_in_names, (data_out_name,))
        self.Lambda = Lambda
        assert 0 <= Lambda and math.isfinite(Lambda)

    def _eval(
        self, beta: ndarray, logp_var: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Lasso_gradinfo_t]]:
        """
        if err[i] ~ N(0,var):
            min 1/n * sum[i](err[i]^2) + L * sum[j](|beta[j]|)

            ~ min sum[i](err[i]^2) + nL * sum[j](|beta[j]|)

            ~ max constant - sum[i]{ err[i]^2 + L*sum[j](|beta[j]|) }

            ~ max constant - sum[i]{ err[i]^2/var[i] + L/var[i]*sum[j](|beta[j]|) }

            ~ max constant - sum[i]{
                                        1/2err[i]^2/var[i]
                                        + L/(2var[i]) * sum[j](|beta[j]|)
                                    }

        while MAP( Laplace priori ):
            log priori[i] = log{ Prod[j]{ 1/(2b) * exp(-|beta[j]|/b) } }
                          = constant - 1/b * sum[j]{|beta[j]|}

            log likeli[i] = log{ 1/sqrt(var[i]*2pi) * exp(-err[i]^2/(2var[i])) } }
                          = constant - 1/2log(var[i]) + 1/2err[i]^2/(var[i])

            log posteriori[i] = constant - {
                                                1/2log(var[i]) + 1/2err[i]^2/(var[i])
                                                + 1/b * sum[j]{|beta[j]|}
                                            }

        so L/(2var[i]) ~ 1/b    =>    b ~ 2var[i]/L

        log priori[i] = log{ Prod[j]{ 1/(2b) * exp(-|beta[j]|/b) } }
                      = -sum[j]{ log(2b) + |beta[j]|/b }
                      = -sum[j]{ log(2 * 2var[i]/L) + |beta[j]|/[2var[i]/L] }
                      = -sum[j]{ log(4)-log(L)+log(var[i]) + L*|beta[j]|/(2var[i]) }
                      = -len(beta)*( log(4)-log(L) )
                        -len(beta)*log(var[i])
                        -L/(2var[i]) * sum[j](|beta[j]|)

        { log(4)-log(L) } ~ constant, but L -> 0 causes constant -> inf, drop it

        log priori[i] = -len(beta)*log(var[i]) - L/(2var[i]) * sum[j](|beta[j]|)
        """
        logP: ndarray = logp_var[:, [0]]
        var: ndarray = logp_var[:, [1]]

        logP = (
            logP
            - beta.shape[0] * numpy.log(var)
            - (self.Lambda / (2.0 * var)) * numpy.sum(numpy.abs(beta))
        )

        if not grad:
            return logP, None
        return logP, (var,)

    def _grad(
        self, beta: ndarray, _var: _Lasso_gradinfo_t, dL_dlogP: ndarray, *, debug: bool
    ) -> Tuple[ndarray, ndarray]:
        """
        d{posteriori[i]}/d{beta}   = -(L/(2var[i])) * sign(beta)
        d{posteriori[i]}/d{var[i]} = -len(beta)/var[i]
                                        + L/2*sum[j](|beta[j]|) / (var[i]*var[i])
        """
        (var,) = _var
        dL_dbeta = -(  # type: ignore
            dL_dlogP.flatten() @ (self.Lambda / (2.0 * var))
        ) * numpy.sign(beta)

        dL_dvar = dL_dlogP * (
            -beta.shape[0] / var
            + (self.Lambda / (2.0)) * numpy.sum(numpy.abs(beta)) / (var * var)
        )
        return (
            numpy.concatenate((dL_dlogP, dL_dvar), axis=1),  # type: ignore
            dL_dbeta,
        )
