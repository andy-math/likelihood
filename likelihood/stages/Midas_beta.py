from typing import Sequence, Tuple

import numpy
from likelihood.KnownIssue import KnownIssue
from likelihood.stages.abc.Convolution import Convolution
from numerical.typedefs import ndarray


class Midas_beta(Convolution):
    K: int

    def __init__(
        self,
        names: Tuple[str, str],
        input: Sequence[int],
        output: Sequence[int],
        *,
        k: int
    ) -> None:
        assert len(input) == len(output)
        super().__init__(names, input, output)
        self.K = k

    def kernel(self, omega: ndarray) -> Tuple[ndarray, ndarray]:
        """
        rphi(1 <= k <= K) = (k/K) ** (omega1-1) * (1-k/K) ** (omega2-1)

        if omega1 <= omega2:
            kLeft, kRight = k, K-k
            oLeft, oRight = omega1-1, omega2-1
        else:
            kLeft, kRight = K-k, k
            oLeft, oRight = omega2-1, omega1-1

        alpha = oLeft / oRight
        stage1 = (kLeft/K) ** alpha * (kRight/K)

        rphi = stage1 ** oRight

        phi = rphi/sum(rphi)
        """
        k = numpy.arange(1.0, self.K + 1.0)
        k.shape = (k.shape[0], 1)
        _omega1, _omega2 = omega

        if _omega1 <= _omega2:
            kLeft, kRight = k, self.K - k
            oLeft, oRight = _omega1 - 1.0, _omega2 - 1.0
        else:
            kLeft, kRight = self.K - k, k
            oLeft, oRight = _omega2 - 1.0, _omega1 - 1.0

        alpha = oLeft / oRight
        da_do = numpy.array([[1.0, -alpha]]) / oRight
        """
        dstage1_da = (kLeft/K) ** alpha * log(kLeft/K) * (1-kRight/K)
                    = stage1 * log(kLeft/K)
                    = rphi * [log(kLeft) - log(K)]
        if kLeft == 0:
            stage1 = 0 ** alpha * (1-kRight/K)
            dstage1_da = 0
        """
        stage1 = (kLeft / self.K) ** alpha * (kRight / self.K)
        dstage1_da = stage1 * (numpy.log(kLeft) - numpy.log(self.K))
        dstage1_da[kLeft == 0] = 0.0
        dstage1_do = dstage1_da * da_do
        """
        rphi = stage1 ** oRight
        drphi_dstage1 = oRight * stage1 ** (oRight-1)
        drphi_doRight = stage1 ** oRight * log(stage1)
                      = rphi * log(stage1)
        if stage1 == 0:
            rphi = 0 ** oRight
            drphi_doRight = 0
        """
        rphi = stage1 ** oRight
        drphi_dstage1 = oRight * stage1 ** (oRight - 1)
        drphi_doRight = rphi * numpy.log(stage1)
        drphi_doRight[stage1 == 0] = 0.0
        drphi_do = drphi_dstage1 * dstage1_do
        drphi_do[:, [1]] += drphi_doRight  # type: ignore

        sum = numpy.sum(rphi)
        dsum_do = numpy.sum(drphi_do, axis=0, keepdims=True)
        if sum == 0:
            raise KnownIssue("Midas_beta: 权重全为0")

        """
        phi = rphi/sum
        dphi_do = (1/sum) * drphi_do - rphi / (sum*sum) * dsum_do
                = drphi_do / sum - dsum_do * phi / sum
        """
        phi = rphi / sum
        dphi_do = drphi_do / sum - dsum_do * phi / sum

        phi.shape = (phi.shape[0],)
        if _omega1 <= _omega2:
            return phi, dphi_do
        else:
            return phi, dphi_do[:, ::-1]

    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        A = numpy.empty((0, 2))
        b = numpy.empty((0,))
        lb = numpy.array([1.0, 1.0])
        ub = numpy.array([numpy.inf, numpy.inf])
        return A, b, lb, ub