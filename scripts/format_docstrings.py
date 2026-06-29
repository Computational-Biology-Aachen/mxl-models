from __future__ import annotations

import ast
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
import typer

app = typer.Typer(add_completion=False)


@dataclass
class FormatResult:
    formatted: str
    changed: bool


def _format_markdown_via_uv(text: str) -> str:
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        msg = "uv executable not found in PATH"
        raise RuntimeError(msg)

    proc = subprocess.run(  # noqa: S603
        [uv_bin, "run", "mdformat", "--wrap", "79", "-"],
        input=text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "mdformat failed"
        raise RuntimeError(stderr)
    return proc.stdout  # .rstrip("\n")


def _to_docstring_literal(text: str) -> cst.SimpleString:
    escaped = text.replace('"""', '\\"\\"\\"')
    return cst.SimpleString(f'"""{escaped}"""')


def _extract_simple_docstring(expr: cst.BaseExpression) -> str | None:
    if isinstance(expr, cst.SimpleString):
        try:
            value = ast.literal_eval(expr.value)
        except (SyntaxError, ValueError):
            return None
        return value if isinstance(value, str) else None
    return None


class DocstringFormatter(cst.CSTTransformer):
    def __init__(self) -> None:
        self.changed = False

    def _rewrite_docstring_stmt(
        self,
        stmt: cst.BaseStatement,
    ) -> cst.BaseStatement:
        if not isinstance(stmt, cst.SimpleStatementLine):
            return stmt
        if len(stmt.body) != 1:
            return stmt

        only = stmt.body[0]
        if not isinstance(only, cst.Expr):
            return stmt

        original_doc = _extract_simple_docstring(only.value)
        if original_doc is None:
            return stmt

        formatted_doc = _format_markdown_via_uv(original_doc)
        if formatted_doc == original_doc:
            return stmt

        self.changed = True
        return stmt.with_changes(
            body=[only.with_changes(value=_to_docstring_literal(formatted_doc))]
        )

    def _rewrite_body(
        self,
        body: tuple[cst.BaseStatement, ...],
    ) -> tuple[cst.BaseStatement, ...]:
        if not body:
            return body

        first = self._rewrite_docstring_stmt(body[0])
        if first is body[0]:
            return body

        return (first, *body[1:])

    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        return updated_node.with_changes(body=self._rewrite_body(updated_node.body))

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        if not isinstance(updated_node.body, cst.IndentedBlock):
            return updated_node
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=self._rewrite_body(updated_node.body.body),
            )
        )

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        if not isinstance(updated_node.body, cst.IndentedBlock):
            return updated_node
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=self._rewrite_body(updated_node.body.body),
            )
        )

    def leave_AsyncFunctionDef(
        self,
        original_node: cst.AsyncFunctionDef,
        updated_node: cst.AsyncFunctionDef,
    ) -> cst.AsyncFunctionDef:
        if not isinstance(updated_node.body, cst.IndentedBlock):
            return updated_node
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=self._rewrite_body(updated_node.body.body),
            )
        )


def _format_file(path: Path) -> FormatResult:
    source = path.read_text(encoding="utf-8")
    module = cst.parse_module(source)
    transformer = DocstringFormatter()
    updated = module.visit(transformer)
    return FormatResult(formatted=updated.code, changed=transformer.changed)


@app.command()
def main(
    paths: list[Path] = typer.Argument(..., exists=True, dir_okay=False),
    check: bool = typer.Option(
        False,
        "--check",
        help="Only check whether formatting changes are needed.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Format Python docstrings in files using mdformat via uv.

    Default behavior writes fixes in-place.
    Use --check for pre-commit/CI validation mode.
    """
    needs_changes: list[Path] = []

    for path in paths:
        if path.suffix != ".py":
            if verbose:
                typer.echo(f"skip (not .py): {path}")
            continue

        try:
            result = _format_file(path)
        except Exception as exc:
            typer.echo(f"error in {path}: {exc}", err=True)
            raise typer.Exit(code=2) from exc

        if not result.changed:
            if verbose:
                typer.echo(f"ok: {path}")
            continue

        needs_changes.append(path)
        if check:
            if verbose:
                typer.echo(f"needs format: {path}")
            continue

        path.write_text(result.formatted, encoding="utf-8")
        typer.echo(f"formatted: {path}")

    if check and needs_changes:
        typer.echo("docstrings need formatting:", err=True)
        for path in needs_changes:
            typer.echo(f"  - {path}", err=True)
        raise typer.Exit(code=1)

    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
