"""Macht jeden `from pytorch3d... import X, Y` / `import pytorch3d...` Top-Level-Import
in hmr4d/ optional (try/except ImportError -> Namen = None). Idempotent (ueberspringt
bereits gepatchte Dateien). Fail-fast bleibt erhalten: wird eine der Funktionen im
tatsaechlich ausgefuehrten Pfad gebraucht, gibt es einen klaren NameError statt eines
Import-Crashs fuer Code, den wir nie durchlaufen (Training/Dataset-Loader, Renderer).

Betrifft NUR reine Rotations-Mathematik (pytorch3d.transforms, kein CUDA) + den
Trainings-Pfad (pytorch3d.ops.knn, pytorch3d.renderer/structures) — letztere bleiben
schlicht ungenutzt fuer den Inferenz-Pfad von demo.py.
"""
import os
import re

ROOT = r"C:\GVHMR\hmr4d"
MARKER = "# [botterdancer-optional-pytorch3d]"

IMPORT_RE = re.compile(
    r"^(from pytorch3d[.\w]* import \([^)]*\)|from pytorch3d[.\w]* import [^\n(][^\n]*"
    r"|import pytorch3d[.\w]*(?:\s+as\s+\w+)?(?:[ \t]*#[^\n]*)?)$",
    re.MULTILINE,
)


def names_from_import(stmt: str):
    # Zeilenend-Kommentare abschneiden, sonst landen sie als "Name" im except-Zweig
    # und erzeugen dort einen SyntaxError (Review 2026-07-19).
    stmt = stmt.split("#", 1)[0].rstrip()
    if stmt.startswith("import "):
        # "import pytorch3d.ops.knn as knn" -> ["knn"]; ohne Alias -> [] -> "pass"
        # (NameError am Use-Site bleibt als fail-fast erhalten)
        m = re.search(r"\bas\s+(\w+)\s*$", stmt)
        return [m.group(1)] if m else []
    # "from pytorch3d.x import (a, b, c)" / "from pytorch3d.x import a, b as c"
    after = stmt.split("import", 1)[1]
    after = after.strip().lstrip("(").rstrip(")")
    names = []
    for n in after.split(","):
        n = n.strip()
        if not n:
            continue
        if " as " in n:
            n = n.split(" as ", 1)[1].strip()  # gebunden wird der Alias, nicht der Originalname
        names.append(n)
    return names


def patch_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # KEIN dateiweiter Marker-Skip mehr (Audit 04.08. F11): eine bereits markierte
    # Datei mit NEUEM pytorch3d-Import wurde sonst komplett uebersprungen.
    # Idempotenz ist strukturell gesichert: gepatchte Imports sind eingerueckt,
    # der ^-Anker des IMPORT_RE kann sie nie erneut matchen.

    def repl(m):
        stmt = m.group(1)
        names = names_from_import(stmt)
        none_assign = " = ".join(names) + " = None" if names else "pass"
        indent = ""
        return (
            f"{MARKER}\ntry:\n{indent}    {stmt}\n"
            f"except ImportError:\n{indent}    {none_assign}"
        )

    new_src, n = IMPORT_RE.subn(repl, src)
    if n == 0:
        return False
    # Syntax-Gate (Re-Attack): der Patcher darf NIE eine Datei mit SyntaxError
    # hinterlassen (z.B. Backslash-Fortsetzungszeilen im Original-Import).
    try:
        compile(new_src, path, "exec")
    except SyntaxError as e:
        print(f"  WARNUNG: Patch fuer {path} erzeugte ungueltiges Python ({e}) — "
              f"Datei bleibt UNVERAENDERT.")
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"  patched {n} import(s) in {path}")
    return True


def find_files():
    hits = []
    for dirpath, _dirnames, filenames in os.walk(ROOT):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if "pytorch3d" in f.read():
                    hits.append(p)
    return hits


if __name__ == "__main__":
    import sys

    files = find_files()
    print(f"{len(files)} Dateien mit pytorch3d-Referenz gefunden.")
    changed = 0
    for f in files:
        if patch_file(f):
            changed += 1
    # Residualscan (Audit 04.08. F11): Imports, die der Regex nicht erfasst hat
    # oder die am Compile-Gate unveraendert blieben, duerfen nicht stumm
    # durchrutschen — ein harter Top-Level-Import braeche spaeter die Inferenz.
    residual = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            if re.search(r"^(from|import)\s+pytorch3d", fh.read(), re.MULTILINE):
                residual.append(f)
    if residual:
        print(f"FEHLER: {len(residual)} Datei(en) mit UNGEPATCHTEN Top-Level-"
              f"pytorch3d-Imports:", file=sys.stderr)
        for f in residual:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)
    print(f"DONE: {changed} Dateien gepatcht, 0 Residual-Imports.")
