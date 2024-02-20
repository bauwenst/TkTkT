from pathlib import Path
import os

PATH_PACKAGE = Path(__file__).resolve().parent.parent
PATH_ROOT    = PATH_PACKAGE.parent.parent  # Can only be accessed if the package was installed in editable mode.

# Output setup: if we can detect that you're running from inside the package project, we assume you don't want CWD.
PATH_OUTPUT = Path(os.getcwd())
if PATH_OUTPUT.is_relative_to(PATH_PACKAGE):  # is_relative_to means "is a descendant of"
    PATH_OUTPUT = PATH_PACKAGE / "data" / "out"

def setTkTkToutputRoot(path: Path):
    global PATH_OUTPUT
    PATH_OUTPUT = path

def getTkTkToutputPath() -> Path:
    out = PATH_OUTPUT / "tktkt"
    out.mkdir(parents=True, exist_ok=True)
    return out
