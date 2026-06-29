import ast
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def get_modules(pkg_dir: Path):
    modules = set()

    for entry in pkg_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name.startswith("_"):
            continue
        if entry.name == "__init__.py":
            continue
        if entry.suffix == ".py":
            modules.add(entry.stem)

    return modules


def extract_imported_modules(init_file: Path):
    if not init_file.exists():
        return set()

    tree = ast.parse(init_file.read_text(encoding="utf-8"))
    imported = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module.lstrip(".")
            imported.add(mod.split(".")[0])

    return imported


def analyze_package(pkg_dir: Path):
    modules = get_modules(pkg_dir)
    imported = extract_imported_modules(pkg_dir / "__init__.py")

    missing = sorted(modules - imported)

    return modules, imported, missing


@app.command()
def check(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Check which modules in a package are not referenced in __init__.py.
    """
    modules, imported, missing = analyze_package(path)

    if verbose:
        typer.echo(f"Modules found: {sorted(modules)}")
        typer.echo(f"Referenced in __init__.py: {sorted(imported)}\n")

    if missing:
        typer.echo("Missing references in __init__.py:")
        for m in missing:
            typer.echo(f"  - {m}")
        raise typer.Exit(code=2)

    typer.echo("All modules are referenced in __init__.py.")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
