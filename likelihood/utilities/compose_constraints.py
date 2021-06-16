from typing import List

import numpy
from likelihood.stages.abc.Stage import Constraints


def compose_constraints(
    expand_mapping: List[int],
    left_constraints: Constraints,
    right_constraints: Constraints,
) -> Constraints:
    from scipy.linalg import block_diag  # type: ignore

    stage1 = Constraints(
        block_diag(left_constraints.A, right_constraints.A),
        numpy.concatenate((left_constraints.b, right_constraints.b)),
        numpy.concatenate((left_constraints.lb, right_constraints.lb)),
        numpy.concatenate((left_constraints.ub, right_constraints.ub)),
    )
    A = numpy.zeros((stage1.A.shape[0], len(expand_mapping)))
    b = stage1.b
    lb = numpy.full((len(expand_mapping),), -numpy.inf)
    ub = numpy.full((len(expand_mapping),), numpy.inf)
    for virtual, actual in enumerate(expand_mapping):
        A[:, actual] += stage1.A[:, virtual]
        lb[actual] = max(lb[actual], stage1.lb[virtual])
        ub[actual] = min(ub[actual], stage1.ub[virtual])
    return Constraints(A, b, lb, ub)
