from __future__ import annotations

from typing import Callable, List, Tuple

from likelihood.jit import Jitted_Function
from likelihood.stages.abc import Iterative
from likelihood.stages.Assign import Assign
from likelihood.stages.Compose import Compose
from likelihood.stages.MS_TVTP import MS_TVTP
from numerical.typedefs import ndarray


class MS_FTP(Compose):
    def __init__(
        _,
        names: Tuple[str, str],
        submodel: Tuple[Iterative.Iterative, Iterative.Iterative],
        sharing: List[str],
        provider: Tuple[
            Jitted_Function[Callable[[ndarray], float]],
            Jitted_Function[Callable[[ndarray, float, float], ndarray]],
        ],
        output: Tuple[int, ...],
    ) -> None:
        assert isinstance(submodel[0], type(submodel[1]))
        assert len(submodel[0]._output_idx) == len(submodel[1]._output_idx)
        assert len(output) - 2 == len(submodel[0]._output_idx)
        stage1 = Assign(names[0], output[-2], 0.0, 1.0)
        stage2 = Assign(names[1], output[-1], 0.0, 1.0)
        stage3 = MS_TVTP(submodel, sharing, provider, (output[-2], output[-1]), output)
        maxinput = max(output)
        super().__init__(
            [stage1, stage2, stage3],
            list(range(maxinput + 1)),
            list(range(maxinput + 1)),
        )
