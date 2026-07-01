from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
import typer

app = typer.Typer(add_completion=False)

# Markdown block openers (tables, lists, headings, blockquotes, code fences)
# that must never be word-wrapped/rejoined as if they were prose.
_BLOCK_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s|[-*+]\s|#{1,6}(?:\s|$)|>|```|\|)")


@dataclass
class FormatResult:
    formatted: str
    changed: bool


@dataclass
class DocstringInfo:
    text: str
    is_raw: bool


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


def _split_indent(text: str) -> tuple[str, str]:
    r"""Strip a docstring's source indentation before markdown formatting.

    Only the first physical line of a docstring value is unindented (it follows the
    opening quotes directly); every later line carries the enclosing block's
    indentation as part of the literal. Left in place, CommonMark reads an indented
    paragraph that follows a blank line as a code block, so a multi-paragraph
    docstring on an indented function or method gets its body wrapped in spurious
    \`\`\` fences. The indentation is captured here and restored by `_apply_indent`
    after formatting.
    """
    lines = text.split("\n")
    if len(lines) <= 1:
        return "", text

    indent = os.path.commonprefix(
        [
            line[: len(line) - len(line.lstrip(" "))]
            for line in lines[1:]
            if line.strip()
        ]
    )
    dedented = [lines[0]]
    dedented.extend(line[len(indent) :] if line.strip() else "" for line in lines[1:])
    return indent, "\n".join(dedented)


_FENCE_LINE_RE = re.compile(r"`{3,}")
_MIN_FENCED_BODY_LINES = 3  # opening fence + >=1 content line + closing fence


def _strip_artifact_fence(text: str) -> str:
    r"""Peel spurious \`\`\` wrapper fences left by the indentation bug.

    Before `_split_indent` existed, an indented docstring's whole body (everything
    after the summary's blank line) was misread by CommonMark as one indented code
    block, and mdformat fenced it off with \`\`\`. Each later run of the buggy
    script then wrapped that fence in one more layer of backticks, since a fence
    can only be closed by a line with at least as many backticks. A fence spanning
    the entire body edge to edge is always that artifact, never an intentionally
    authored code block, so it's peeled off (possibly several nested layers) before
    reformatting.
    """
    lines = text.split("\n")
    try:
        blank_idx = lines.index("")
    except ValueError:
        return text

    head, body = lines[: blank_idx + 1], lines[blank_idx + 1 :]
    trailing: list[str] = []
    while body and body[-1] == "":
        trailing.insert(0, body.pop())

    while (
        len(body) >= _MIN_FENCED_BODY_LINES
        and _FENCE_LINE_RE.fullmatch(body[0].strip())
        and _FENCE_LINE_RE.fullmatch(body[-1].strip())
    ):
        body = body[1:-1]

    return "\n".join([*head, *body, *trailing])


def _apply_indent(text: str, indent: str) -> str:
    # mdformat always appends a trailing newline, even to single-line input.
    # If the formatted content is genuinely one line, collapse it back to a
    # true one-liner (pydocstyle D200) instead of indenting a bogus closing
    # line; the resulting docstring literal ends up all on one source line
    # regardless of the enclosing block's indentation.
    if text.count("\n") == 1 and text.endswith("\n"):
        return text[:-1]

    if not indent:
        return text
    lines = text.split("\n")
    last = len(lines) - 1
    for i in range(1, len(lines)):
        if i == last or lines[i]:
            lines[i] = indent + lines[i]
    return "\n".join(lines)


def _unwrap_first_paragraph(formatted: str) -> str:
    r"""Undo mdformat's word-wrap on the docstring's first paragraph.

    pydocstyle's numpy convention requires a blank line immediately after the
    summary line. If the summary is longer than the wrap width, mdformat folds it
    across multiple physical lines with no blank line in between, which trips that
    rule. The first paragraph is rejoined into a single line (however long) so the
    wrap width only applies to the rest of the docstring.
    """
    lines = formatted.split("\n")
    end = 0
    while end < len(lines) and lines[end].strip() != "":
        end += 1

    first_block = lines[:end]
    if len(first_block) <= 1:
        return formatted
    if any(_BLOCK_MARKER_RE.match(line) for line in first_block):
        return formatted

    joined = " ".join(line.strip() for line in first_block)
    return "\n".join([joined, *lines[end:]])


def _to_docstring_literal(text: str) -> cst.SimpleString:
    if '"""' in text:
        msg = 'docstring contains a literal triple-quote; cannot emit as r"""'
        raise RuntimeError(msg)
    return cst.SimpleString(f'r"""{text}"""')


def _extract_simple_docstring(expr: cst.BaseExpression) -> DocstringInfo | None:
    if not isinstance(expr, cst.SimpleString):
        return None
    value = expr.evaluated_value
    if not isinstance(value, str):
        return None
    return DocstringInfo(text=value, is_raw=expr.prefix.lower() == "r")


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

        indent, dedented = _split_indent(original_doc.text)
        if indent:
            dedented = _strip_artifact_fence(dedented)
        formatted_doc = _format_markdown_via_uv(dedented)
        formatted_doc = _unwrap_first_paragraph(formatted_doc)
        formatted_doc = _apply_indent(formatted_doc, indent)
        if formatted_doc == original_doc.text and original_doc.is_raw:
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
    r"""Format Python docstrings in files using mdformat via uv.

    Default behavior writes fixes in-place. Use --check for pre-commit/CI
    validation mode.
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
