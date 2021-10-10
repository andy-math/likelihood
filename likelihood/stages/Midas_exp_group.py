from typing import Callable, Optional, Tuple

from mypy_extensions import NamedArg

from likelihood.stages.abc.Stage import Constraints, Stage
from likelihood.stages.Linear import Linear
from likelihood.stages.Midas_exp import Midas_exp
from overloads.typedefs import ndarray

_Midas_exp_group_gradinfo_t = Tuple[ndarray, ndarray, ndarray]


class Midas_exp_group(Stage[_Midas_exp_group_gradinfo_t]):
    _linear_eval: Optional[
        Callable[
            [
                ndarray,
                ndarray,
                NamedArg(bool, "grad"),  # noqa: F821
                NamedArg(bool, "debug"),  # noqa: F821
            ],
            Tuple[ndarray, Optional[ndarray]],
        ]
    ]
    _linear_grad: Optional[
        Callable[
            [ndarray, ndarray, ndarray, NamedArg(bool, "debug")],  # noqa: F821
            Tuple[ndarray, ndarray],
        ]
    ]
    _kernel: Optional[Callable[[ndarray], Tuple[ndarray, ndarray]]]
    _constraints: Optional[Callable[[], Constraints]]

    def __init__(
        self, coeff_name: str, data_in_names: Tuple[str, ...], data_out_names: str
    ) -> None:
        super().__init__((coeff_name,), data_in_names, (data_out_names,), ())
        linear_obj = Linear((coeff_name,), data_in_names, data_out_names)
        self._linear_eval = linear_obj._eval
        self._linear_grad = linear_obj._grad
        midas_obj = Midas_exp(
            coeff_name, data_in_names, data_in_names, k=len(data_in_names)
        )
        self._kernel = midas_obj.kernel
        self._constraints = midas_obj.get_constraints

    def _eval(
        self, coeff: ndarray, input: ndarray, *, grad: bool, debug: bool
    ) -> Tuple[ndarray, Optional[_Midas_exp_group_gradinfo_t]]:
        assert self._kernel is not None
        assert self._linear_eval is not None
        kernel, dk_dc = self._kernel(coeff)
        output, _gradinfo = self._linear_eval(kernel, input, grad=grad, debug=debug)
        if not grad:
            return output, None
        assert _gradinfo is not None
        return output, (_gradinfo, kernel, dk_dc)

    def _grad(
        self,
        coeff: ndarray,
        gradinfo: _Midas_exp_group_gradinfo_t,
        dL_do: ndarray,
        *,
        debug: bool
    ) -> Tuple[ndarray, ndarray]:
        assert self._linear_grad is not None
        _gradinfo, kernel, dk_dc = gradinfo
        dL_di, dL_dk = self._linear_grad(kernel, _gradinfo, dL_do, debug=debug)
        return dL_di, dL_dk @ dk_dc

    def get_constraints(self) -> Constraints:
        assert self._constraints is not None
        return self._constraints()
