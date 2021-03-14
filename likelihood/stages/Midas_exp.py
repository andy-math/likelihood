from typing import Sequence, Tuple

import numpy
from likelihood.stages.abc.Convolution import Convolution
from numerical.typedefs import ndarray


class Midas_exp(Convolution):
    k: int

    def __init__(
        self, names: str, input: Sequence[int], output: Sequence[int], *, k: int
    ) -> None:
        assert len(input) == len(output)
        super().__init__((names,), input, output)
        self.k = k

    def kernel(self, omega: ndarray) -> Tuple[ndarray, ndarray]:
        """
        rphi(1<= k <= K) = omega ** k
        phi = rphi/sum(rphi)
        0 < omega < 1
        """
        k = numpy.arange(1.0, self.k + 1.0)

        rphi = omega ** k
        drphi_do = k * omega ** (k - 1.0)

        if max(rphi) == 0:
            assert False

        sum = numpy.sum(rphi)
        dsum_do = numpy.sum(drphi_do)

        phi = rphi / sum
        dphi_do = drphi_do / sum - rphi * dsum_do / (sum * sum)

        dphi_do.shape = (dphi_do.shape[0], 1)
        return phi, dphi_do

    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        A = numpy.empty((0, 1))
        b = numpy.empty((0,))
        lb = numpy.array([0.0])
        ub = numpy.array([1.0])
        return A, b, lb, ub
