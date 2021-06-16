from typing import List, Tuple


def compose_names(
    shared_names: Tuple[str, ...],
    left_names: Tuple[str, ...],
    right_names: Tuple[str, ...],
) -> Tuple[Tuple[str, ...], List[int]]:
    """
    组合两个模型的参数名称并去除多余的重名（共享）参数
    输出组合的完整参数列表（Tuple）和展开索引式idx: tup[idx] => [left, right]
    """
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
