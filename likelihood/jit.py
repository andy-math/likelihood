from __future__ import annotations

import multiprocessing
import pickle
import time
from typing import Any, Callable, Dict, Generic, NoReturn, Tuple, TypeVar

import numba  # type: ignore

T2 = TypeVar("T2", covariant=True)

_output_width_m = 0
_output_width_n = 0
_Jitted_Function_Cache: Dict[Tuple[bytes, ...], Tuple[Any, Any]] = {}


class Jitted_Function(Generic[T2]):
    signature: numba.core.typing.templates.Signature
    dependent: Tuple[Jitted_Function[Any], ...]
    pickled_bytecode: Tuple[bytes, ...]

    def __init__(
        self,
        signature: numba.core.typing.templates.Signature,
        dependent: Tuple[Jitted_Function[Any], ...],
        generator: Callable[..., T2],
    ) -> None:
        global _output_width_m, _output_width_n
        # picklable test
        self.pickled_bytecode = (pickle.dumps(generator),) + tuple(
            [y for x in dependent for y in x.pickled_bytecode]
        )
        self.signature = signature
        self.dependent = dependent
        _output_width_m = max(_output_width_m, len(generator.__module__))
        _output_width_n = max(_output_width_n, len(generator.__name__))

    def _compile(self) -> Tuple[T2, T2]:
        if self.pickled_bytecode in _Jitted_Function_Cache:
            return _Jitted_Function_Cache[self.pickled_bytecode]

        start_time = time.time()

        generator: Callable[..., T2] = pickle.loads(self.pickled_bytecode[0])

        func = numba.njit(self.signature)(
            generator(*[x.func() for x in self.dependent])
        )
        py_func = generator(*[x.py_func() for x in self.dependent])

        _Jitted_Function_Cache[self.pickled_bytecode] = (func, py_func)
        print(
            f"pid[{multiprocessing.current_process().pid}]: "
            f"预编译 {generator.__module__.ljust(_output_width_m+6)} "
            f".{generator.__name__.ljust(_output_width_n+6)} 完成"
            f" -- 用时{time.time()-start_time:.4f}秒"
        )
        return func, py_func

    def func(self) -> T2:
        func, _ = self._compile()
        return func

    def py_func(self) -> T2:
        _, py_func = self._compile()
        return py_func

    def __call__(_) -> NoReturn:
        assert False  # pragma: no cover
