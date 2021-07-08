from __future__ import annotations

import ast
import builtins
import os
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

from mypy_extensions import VarArg

saveload_t = Union[Type[ast.Load], Type[ast.Store]]


def extractExprRef(
    expr: Union[ast.expr, ast.alias], add: Callable[[str], None], saveload: saveload_t
) -> None:
    def call_self(*_exprs: Optional[ast.expr], sl: Optional[saveload_t] = None) -> None:
        for e in _exprs:
            if e is not None:
                extractExprRef(e, add, saveload if sl is None else sl)

    if hasattr(expr, "ctx"):
        assert isinstance(expr.ctx, saveload)  # type: ignore
    if isinstance(expr, ast.Constant):
        pass  # drop
    elif isinstance(expr, ast.Name):
        add(expr.id)
    elif isinstance(expr, ast.alias):
        add(expr.name if expr.asname is None else expr.asname)
    elif isinstance(expr, ast.Subscript):
        call_self(expr.value, expr.slice, sl=ast.Load)
    elif isinstance(expr, (ast.List, ast.Tuple)):
        call_self(*expr.elts)
    elif isinstance(expr, ast.Call):
        call_self(expr.func, *expr.args)
    elif isinstance(expr, ast.Attribute):
        call_self(expr.value)
    elif isinstance(expr, ast.Slice):
        call_self(expr.lower, expr.upper, expr.step)
    elif isinstance(expr, ast.BinOp):
        call_self(expr.left, expr.right)
    elif isinstance(expr, ast.UnaryOp):
        call_self(expr.operand)
    elif isinstance(expr, ast.Compare):
        call_self(expr.left, *expr.comparators)
    elif isinstance(expr, ast.IfExp):
        call_self(expr.test, expr.body, expr.orelse)
    elif isinstance(expr, ast.BoolOp):
        call_self(*expr.values)
    elif isinstance(expr, ast.Dict):
        call_self(*expr.keys, *expr.values)
    else:
        assert False, f"未被考虑到的表达式类型{type(expr).__name__}\n{ast.unparse(expr)}"


def extractFuncRef(
    function: ast.FunctionDef,
    defined: Set[str],
    reference: Callable[[VarArg(Optional[ast.expr])], None],
) -> Set[str]:
    assert len(function.decorator_list) == 0
    args = function.args
    assert len(args.posonlyargs) == 0
    assert args.vararg is None
    assert len(args.kwonlyargs) == 0
    assert len(args.kw_defaults) == 0
    assert args.kwarg is None
    assert len(args.defaults) == 0
    reference(*(arg.annotation for arg in args.args), function.returns)
    defined = defined.copy()
    defined.update({arg.arg for arg in args.args})
    undefined: Set[str] = set()
    extractStmtRef(function.body, defined, undefined)
    return undefined


def extractStmtRef(
    statements: List[ast.stmt], defined: Set[str], undefined: Set[str]
) -> None:
    declared = False

    def add_undefined(x: str) -> None:
        if x not in builtins.__dict__ and x not in defined:
            undefined.add(x)

    def reference(*exprs: Optional[ast.expr]) -> None:
        assert not declared
        for e in exprs:
            if e is not None:
                extractExprRef(e, add_undefined, ast.Load)

    def declare(*exprs: Optional[Union[ast.expr, ast.alias]]) -> None:
        nonlocal declared
        for e in exprs:
            if e is not None:
                extractExprRef(e, defined.add, ast.Store)
        declared = True

    for stmt in statements:
        declared = False
        assert all(x not in defined for x in undefined)
        if isinstance(stmt, ast.Assign):
            reference(stmt.value)
            declare(*stmt.targets)
        elif isinstance(stmt, ast.AugAssign):
            reference(stmt.value)
            declare(stmt.target)
        elif isinstance(stmt, ast.AnnAssign):
            reference(stmt.value, stmt.annotation)
            declare(stmt.target)
        elif isinstance(stmt, ast.FunctionDef):
            undefined.update(extractFuncRef(stmt, defined, reference))
            defined.add(stmt.name)
        elif isinstance(stmt, ast.expr):
            reference(stmt)
        elif isinstance(stmt, ast.For):
            assert len(stmt.orelse) == 0
            reference(stmt.iter)
            declare(stmt.target)
            extractStmtRef(stmt.body, defined, undefined)
        elif isinstance(stmt, ast.If):
            reference(stmt.test)
            unpatched_def = defined.copy()
            extractStmtRef(stmt.body, defined, undefined)
            extractStmtRef(stmt.body, unpatched_def, undefined)
            defined.update(unpatched_def)
        elif isinstance(stmt, ast.Assert):
            reference(stmt.test)
        elif isinstance(stmt, ast.Return):
            reference(stmt.value)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            declare(*stmt.names)
        elif isinstance(stmt, ast.Expr):
            reference(stmt.value)
        else:
            assert False, f"未被考虑到的语句类型{type(stmt).__name__}\n{ast.unparse(stmt)}"
    assert all(x not in defined for x in undefined)


def extractStmtRef_(*stmt: ast.stmt) -> Tuple[Set[str], Set[str]]:
    defined: Set[str] = set()
    undefined: Set[str] = set()
    extractStmtRef(list(stmt), defined, undefined)
    return defined, undefined


global_stmt_t = Union[
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
]


def using_symbol(
    global_stmt: List[global_stmt_t], funcDef: ast.FunctionDef
) -> List[ast.stmt]:
    # 导出全局命名空间中的符号，import的和assign的
    symbols: Dict[global_stmt_t, Set[str]] = {}
    for stmt in global_stmt:
        symbols[stmt], _ = extractStmtRef_(stmt)

    # 将用到的符号相关的语句放到集合里
    using: Set[global_stmt_t] = {funcDef}
    for _ in range(len(global_stmt)):
        _, undefined = extractStmtRef_(
            *(stmt for stmt in global_stmt if stmt in using), funcDef
        )
        if len(undefined) == 0:
            break
        for stmt in global_stmt:
            if stmt not in using and any(x in undefined for x in symbols[stmt]):
                using.add(stmt)
    assert len(undefined) == 0
    return [stmt for stmt in global_stmt if stmt in using]


def split_statements(
    statements: List[ast.stmt],
) -> Tuple[
    List[Union[ast.Import, ast.ImportFrom]],
    List[Union[ast.Assign, ast.AugAssign, ast.AnnAssign]],
    ast.FunctionDef,
]:
    imports: List[Union[ast.Import, ast.ImportFrom]] = []
    assigns: List[Union[ast.Assign, ast.AugAssign, ast.AnnAssign]] = []
    funcdef: Optional[ast.FunctionDef]
    for i, stmt in enumerate(statements):
        if i == len(statements) - 1:
            assert isinstance(stmt, ast.FunctionDef)
            funcdef = stmt
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imports.append(stmt)
        elif isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            assigns.append(stmt)
        else:
            assert False
    assert funcdef is not None
    return imports, assigns, funcdef


def prune(tree: ast.Module, entry_name: str) -> None:
    # 展开所有import
    class ExpandImport(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> Union[ast.AST, List[ast.stmt]]:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                statements: List[ast.stmt] = []
                for name in node.names:
                    shadow = deepcopy(node)
                    shadow.names = [name]
                    statements.append(shadow)
                return statements
            else:
                return node

    ExpandImport().generic_visit(tree)

    # 寻找函数入口
    global_stmt: List[global_stmt_t] = []
    entryDef: Optional[ast.FunctionDef] = None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == entry_name:
            entryDef = stmt
        if isinstance(
            stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Import, ast.ImportFrom)
        ):
            global_stmt.append(stmt)
    assert entryDef is not None, f"函数入口{entry_name}未找到"
    tree.body = [*using_symbol(global_stmt, entryDef), entryDef]


def patch(filename: str, entry_name: str) -> Generator:
    with open(filename) as f:
        str_code = f.read()
    tree = ast.parse(str_code, filename=filename)
    prune(tree, entry_name)
    prefix = "".join(filename.replace(".py", "").split(os.sep))
    prefix = prefix + "__" + entry_name

    class ReName(ast.NodeTransformer):
        def _rename(self, name: str) -> str:
            if name in builtins.__dict__:
                return name
            name = prefix + "__" + name
            return name

        def visit(self, node: ast.AST) -> ast.AST:
            if isinstance(node, ast.Name):
                node.id = self._rename(node.id)
            elif isinstance(node, ast.FunctionDef):
                node.name = self._rename(node.name)
            elif isinstance(node, ast.arg):
                node.arg = self._rename(node.arg)
            elif isinstance(node, ast.alias):
                node.asname = self._rename(
                    node.name if node.asname is None else node.asname
                )
            self.generic_visit(node)
            return node

    ReName().generic_visit(tree)

    return Generator(tree)


class InnerFunction:
    imports: List[Union[ast.Import, ast.ImportFrom]]
    assigns: List[Union[ast.Assign, ast.AugAssign, ast.AnnAssign]]
    funcdef: ast.FunctionDef

    def __init__(
        self,
        imports: List[Union[ast.Import, ast.ImportFrom]],
        assigns: List[Union[ast.Assign, ast.AugAssign, ast.AnnAssign]],
        funcdef: ast.FunctionDef,
    ) -> None:
        self.imports = imports
        self.assigns = assigns
        self.funcdef = funcdef
        assert len(funcdef.args.posonlyargs) == 0
        assert funcdef.args.vararg is None
        assert len(funcdef.args.kwonlyargs) == 0
        assert len(funcdef.args.kw_defaults) == 0
        assert funcdef.args.kwarg is None
        assert len(funcdef.args.defaults) == 0

    def expand(self, input: List[ast.expr]) -> Tuple[List[ast.stmt], ast.expr]:
        stmts = deepcopy(self.funcdef.body)
        # 内层函数中只能有一个return
        return_stmt = stmts.pop(-1)
        assert isinstance(return_stmt, ast.Return)
        Generator._assertNoReturn(stmts)
        # 根据input生成assign
        assert len(input) == len(self.funcdef.args.args)
        stmts = [
            ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id=arg.arg, ctx=ast.Store())
                            for arg in self.funcdef.args.args
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Tuple(
                    elts=input,
                    ctx=ast.Load(),
                ),
            ),
            *stmts,
        ]

        assert return_stmt.value is not None
        return stmts, return_stmt.value

    def __str__(self) -> str:
        module = ast.Module(
            body=[*self.imports, *self.assigns, self.funcdef], type_ignores=[]
        )
        ast.fix_missing_locations(module)
        return ast.unparse(module)


class Generator:
    tree: ast.Module

    @staticmethod
    def _assertNoReturn(statements: List[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.For):
                Generator._assertNoReturn(stmt.body)
                Generator._assertNoReturn(stmt.orelse)
            elif isinstance(stmt, ast.If):
                Generator._assertNoReturn(stmt.body)
                Generator._assertNoReturn(stmt.orelse)
            else:
                assert not isinstance(stmt, ast.Return)

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        # 只有一个函数声明，且函数声明在最底下
        func = tree.body[-1]
        assert isinstance(func, ast.FunctionDef)
        assert not any(isinstance(x, ast.FunctionDef) for x in tree.body[:-1])
        assert len(func.args.posonlyargs) == 0
        assert func.args.vararg is None
        assert len(func.args.kwonlyargs) == 0
        assert len(func.args.kw_defaults) == 0
        assert func.args.kwarg is None
        assert len(func.args.defaults) == 0
        # 其它要么是import，要么是assign，且import在assign前面
        meet_assign = False
        for x in tree.body[:-1]:
            if isinstance(x, (ast.Import, ast.ImportFrom)):
                assert not meet_assign
            elif isinstance(x, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                meet_assign = True
            else:
                assert False
        # 函数里只有两句，第一句是函数声明，第二句是返回这一函数
        assert len(func.body) == 2
        assert isinstance(func.body[0], ast.FunctionDef)
        assert len(func.body[0].args.posonlyargs) == 0
        assert func.body[0].args.vararg is None
        assert len(func.body[0].args.kwonlyargs) == 0
        assert len(func.body[0].args.kw_defaults) == 0
        assert func.body[0].args.kwarg is None
        assert len(func.body[0].args.defaults) == 0
        assert isinstance(func.body[1], ast.Return)
        assert isinstance(func.body[1].value, ast.Name)
        assert func.body[1].value.id == func.body[0].name

    def substitute(self, *_input: InnerFunction) -> InnerFunction:
        tree = deepcopy(self.tree)
        func = tree.body[-1]
        assert isinstance(func, ast.FunctionDef)
        args = func.args
        assert len(args.args) == len(_input)
        # 构建函数参数名到输入函数之间的映射
        input: Dict[str, InnerFunction] = {
            arg.arg: f for arg, f in zip(args.args, _input)
        }
        used: Dict[str, bool] = {name: False for name in input}

        # 对tree做遍历，遇到function call到input的地方，对input展开并插入

        class Sub(ast.NodeTransformer):
            def visit(self, node: ast.AST) -> Union[ast.AST, List[ast.stmt]]:
                def justreturn() -> ast.AST:
                    self.generic_visit(node)
                    return node

                if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    return justreturn()
                if node.value is None:
                    return justreturn()
                if not isinstance(node.value, ast.Call):
                    return justreturn()
                if not isinstance(node.value.func, ast.Name):
                    return justreturn()
                if node.value.func.id not in input:
                    return justreturn()
                used[node.value.func.id] = True
                assert len(node.value.keywords) == 0
                stmts, value = input[node.value.func.id].expand(node.value.args)
                node.value = value
                stmts.append(node)
                return stmts

        Sub().generic_visit(tree)
        assert not len(used) or all(used.values())
        imports, assigns, funcdef = split_statements(tree.body)
        # 插入import和assign
        for x in input.values():
            imports += x.imports
            assigns += x.assigns
        # 对外层函数做unwarp
        assert isinstance(funcdef.body[0], ast.FunctionDef)
        funcdef = funcdef.body[0]
        _, undefined = extractStmtRef_(*imports, *assigns, funcdef)
        assert not len(undefined)
        return InnerFunction(
            *split_statements([*using_symbol([*imports, *assigns], funcdef), funcdef])
        )

    def __str__(self) -> str:
        return ast.unparse(self.tree)


def h(g: Callable[[float], float]) -> Callable[[float], float]:
    def f(x: float) -> float:
        y = g(x)
        return 2 * y

    return f


def g_generator() -> Callable[[float], float]:
    def g(x: float) -> float:
        return 2 * x

    return g


if __name__ == "__main__":
    g_ = patch("trial.py", "g_generator").substitute()
    f_ = patch("trial.py", "h").substitute(g_)
    print(f_)
    """
    normpdf_provider = patch(
        "likelihood/stages/MS_TVTP.py", "normpdf_provider"
    ).substitute()
    _iterize_eval_generate = patch(
        "likelihood/stages/Iterize.py", "_iterize_eval_generate"
    ).substitute()
    _iterize_output0_generate_3 = patch(
        "likelihood/stages/Iterize.py", "_iterize_output0_generate_3"
    ).substitute()
    _tvtp_eval_generate = patch(
        "likelihood/stages/MS_TVTP.py", "_tvtp_eval_generate"
    ).substitute(_iterize_eval_generate, _iterize_eval_generate, normpdf_provider)
    _tvtp_output0_generate = patch(
        "likelihood/stages/MS_TVTP.py", "_tvtp_output0_generate"
    ).substitute(_iterize_output0_generate_3, _iterize_output0_generate_3)
    _eval_generator = patch(
        "likelihood/stages/abc/Iterative.py", "_eval_generator"
    ).substitute(_tvtp_output0_generate, _tvtp_eval_generate)
    with open("trial2.py", "w") as f:
        f.write(str(_eval_generator))
    """
