from __future__ import annotations

import multiprocessing
import os
import pickle
import tempfile
import time
from typing import Any, Callable, Dict, Generic, NoReturn, Optional, Tuple, TypeVar

import numba  # type: ignore

from likelihood.substitute import InnerFunction, patch

T2 = TypeVar("T2", covariant=True)

_output_width_m = 0
_output_width_n = 0
_Jitted_Function_Cache: Dict[Tuple[bytes, ...], Tuple[Any, Any]] = {}


def load_func(innerfunc: InnerFunction) -> Any:
    import importlib.util
    import sys

    fname = innerfunc.get_name()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".py", delete=False
    ) as f:
        f.write(str(innerfunc))
        filename = f.name
    spec = importlib.util.spec_from_file_location(fname, filename)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[fname] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    return module.__dict__[fname]


class Jitted_Function(Generic[T2]):
    signature: numba.core.typing.templates.Signature
    dependent: Tuple[Jitted_Function[Any], ...]
    innerfunc: InnerFunction
    pickled_bytecode: Tuple[bytes, ...]

    def __init__(
        self,
        signature: numba.core.typing.templates.Signature,
        dependent: Tuple[Jitted_Function[Any], ...],
        generator: Callable[..., T2],
    ) -> None:
        # picklable test
        pickled_bytecode = tuple(
            [
                pickle.dumps(generator),
                *(y for x in dependent for y in x.pickled_bytecode),
            ]
        )

        # 代换编译
        innerfunc = patch(
            generator.__module__.replace(".", os.sep) + ".py", generator.__name__
        ).substitute(*(dep.innerfunc for dep in dependent))
        print(f"展开代换 {generator.__module__}.{generator.__name__} 完成")

        self.__setstate__((signature, dependent, innerfunc, pickled_bytecode))

    def __getstate__(
        self,
    ) -> Tuple[
        numba.core.typing.templates.Signature,
        Tuple[Jitted_Function[Any], ...],
        InnerFunction,
        Tuple[bytes, ...],
    ]:
        return (
            self.signature,
            self.dependent,
            self.innerfunc,
            self.pickled_bytecode,
        )

    def __setstate__(
        self,
        state: Tuple[
            numba.core.typing.templates.Signature,
            Tuple[Jitted_Function[Any], ...],
            InnerFunction,
            Tuple[bytes, ...],
        ],
    ) -> None:
        global _output_width_m, _output_width_n
        (
            self.signature,
            self.dependent,
            self.innerfunc,
            self.pickled_bytecode,
        ) = state
        generator = self._get_generator()
        _output_width_m = max(_output_width_m, len(generator.__module__))
        _output_width_n = max(_output_width_n, len(generator.__name__))

    def _get_generator(self) -> Callable[..., T2]:
        return pickle.loads(self.pickled_bytecode[0])  # type: ignore

    def _compile(self, compile: bool) -> Tuple[Optional[T2], T2]:
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
            # load_func(self.innerfunc)
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

    def func(self) -> T2:
        func, _ = self._compile(True)
        assert func is not None
        return func

    def py_func(self) -> T2:
        _, py_func = self._compile(False)
        return py_func

    def __call__(_) -> NoReturn:
        assert False  # pragma: no cover
