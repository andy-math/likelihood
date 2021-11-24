from __future__ import annotations

import multiprocessing
import pickle
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    NewType,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
)

import numba  # type: ignore

_signature_t = NewType("_signature_t", object)
_function_t = TypeVar("_function_t", covariant=True)

_output_width_m = 0
_output_width_n = 0
_Jitted_Function_Cache: Dict[Tuple[bytes, ...], Tuple[Any, Any]] = {}


class JittedFunction(Generic[_function_t]):
    signature: _signature_t
    pickled_bytecode: Tuple[bytes, ...]
    dependent: Tuple[JittedFunction[Any], ...]

    def __init__(
        self,
        signature: _signature_t,
        dependent: Tuple[JittedFunction[Any], ...],
        generator: Callable[..., _function_t],
    ) -> None:
        # picklable test
        pickled_bytecode = (
            pickle.dumps(generator),
            *(y for x in dependent for y in x.pickled_bytecode),
        )

        self.__setstate__((signature, pickled_bytecode, dependent))

    def __getstate__(
        self,
    ) -> Tuple[_signature_t, Tuple[bytes, ...], Tuple[JittedFunction[Any], ...]]:
        return (self.signature, self.pickled_bytecode, self.dependent)

    def __setstate__(
        self,
        state: Tuple[_signature_t, Tuple[bytes, ...], Tuple[JittedFunction[Any], ...]],
    ) -> None:
        global _output_width_m, _output_width_n
        (self.signature, self.pickled_bytecode, self.dependent) = state
        generator = self._get_generator()
        _output_width_m = max(_output_width_m, len(generator.__module__))
        _output_width_n = max(_output_width_n, len(generator.__name__))

    def _get_generator(self) -> Callable[..., _function_t]:
        return pickle.loads(self.pickled_bytecode[0])  # type: ignore

    def _compile(self, compile: bool) -> Tuple[Optional[_function_t], _function_t]:
        if self.pickled_bytecode in _Jitted_Function_Cache:
            return _Jitted_Function_Cache[self.pickled_bytecode]

        generator = self._get_generator()
        py_func = generator(*(x.py_func() for x in self.dependent))
        if not compile:
            return None, py_func

        print(
            f"pid[{multiprocessing.current_process().pid}]: "
            f"预编译 {generator.__module__.ljust(_output_width_m)} "
            f".{generator.__name__.ljust(_output_width_n)} 等候中\n",
            end="",
        )

        start_time = time.time()
        func = numba.njit(self.signature)(
            generator(*(x.func() for x in self.dependent))
        )
        _Jitted_Function_Cache[self.pickled_bytecode] = (func, py_func)

        print(
            f"pid[{multiprocessing.current_process().pid}]: "
            f"预编译 {generator.__module__.ljust(_output_width_m)} "
            f".{generator.__name__.ljust(_output_width_n)} 完成"
            f" -- 用时{time.time()-start_time:.4f}秒\n",
            end="",
        )

        return func, py_func

    def func(self) -> _function_t:
        func, _ = self._compile(True)
        assert func is not None
        return func

    def py_func(self) -> _function_t:
        _, py_func = self._compile(False)
        return py_func

    def __call__(_) -> NoReturn:
        assert False  # pragma: no cover
