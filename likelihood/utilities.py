from typing import List, Tuple

import numpy

from likelihood.stages.abc.Stage import Constraints


def compose_names(
    shared_names: Tuple[str, ...],
    left_names: Tuple[str, ...],
    right_names: Tuple[str, ...],
) -> Tuple[Tuple[str, ...], List[int]]:
    # names没有内部重名
    assert len(shared_names) == len(set(shared_names))
    assert len(left_names) == len(set(left_names))
    assert len(right_names) == len(set(right_names))

    for name in shared_names:  # sharing的内容同时出现于两端
        assert name in left_names and name in right_names
    for name in left_names:  # left_names的东西在sharing里或者不在right_names里
        assert name in shared_names or name not in right_names
    for name in right_names:  # 右侧同理
        assert name in shared_names or name not in left_names

    # 合成名称列表原则：优先照顾左边，再加入左边没有的右边
    compose_names: List[str] = []
    # 扩张映射原则：compose_names[expand] == [left, right]
    expand_mapping: List[int] = []
    for index, name in enumerate(left_names):
        compose_names.append(name)
        expand_mapping.append(index)
    for index, name in enumerate(right_names):
        if name not in shared_names:
            compose_names.append(name)
            expand_mapping.append(len(compose_names) - 1)
        else:
            expand_mapping.append(compose_names.index(name))
    return tuple(compose_names), expand_mapping


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
