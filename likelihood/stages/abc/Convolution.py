from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import numpy
from likelihood.stages.abc.Stage import Stage
from numerical.typedefs import ndarray

_Convolution_gradinfo_t = Tuple[ndarray, ndarray, ndarray]


class Convolution(Stage[_Convolution_gradinfo_t], metaclass=ABCMeta):
    @abstractmethod
    def kernel(self, coeff: ndarray) -> Tuple[ndarray, ndarray]:
        pass  # pragma: no cover

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool
    ) -> Tuple[ndarray, Optional[_Convolution_gradinfo_t]]:
        """
        out[i] = in[i]*ker[0] + in[i-1]*ker[1] + ... + in[i-k]*ker[k]

        numpy.convolve:
            conv(in,k)[i] = sum(in[i-j]*ker[j])
        """
        kernel, dk_dc = self.kernel(coeff)
        k = kernel.shape[0] - 1
        assert input.shape[0] > k
        output = numpy.empty((input.shape[0] - k, input.shape[1]))
        for i in range(input.shape[1]):
            output[:, i] = numpy.convolve(input[:, i], kernel, "valid")  # type: ignore
        if not grad:
            return output, None
        return output, (input, kernel, dk_dc)

    def _grad(
        self, coeff: ndarray, gradinfo: _Convolution_gradinfo_t, dL_do: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """
        dL_di[i] = dL_do[i]*ker[0] + dL_do[i+1]*ker[1] + ... + dL_do[i+k]*ker[k]
                 = dL_do[i+k]*ker[k] + dL_do[i+k-1]*ker[k-1] + ... + dL_do[i]*ker[0]
        dL_dk[j] = in[k-j]*dL_do[0] + in[k-j+1]*dL_do[1] + ... + in[n-j]*dL_do[n-k]
        """
        input, kernel, dk_dc = gradinfo
        input, kernel = input[::-1, :], kernel[::-1]
        k = kernel.shape[0] - 1
        dL_di = numpy.empty((dL_do.shape[0] + k, dL_do.shape[1]))
        dL_dk = numpy.zeros(kernel.shape)
        for i in range(dL_di.shape[1]):
            dL_di[:, i] = numpy.convolve(dL_do[:, i], kernel, "full")  # type: ignore
            dL_dk += numpy.convolve(input[:, i], dL_do[:, i], "valid")  # type: ignore
        return dL_di, dL_dk @ dk_dc
