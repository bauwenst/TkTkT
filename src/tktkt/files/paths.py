from pathlib import Path
import os

PATH_PACKAGE = Path(__file__).resolve().parent.parent
PATH_ROOT    = PATH_PACKAGE.parent.parent  # Can only be accessed if the package was installed in editable mode.

PATH_OUTPUT = Path(os.getcwd())
def setTkTkToutputRoot(path: Path):
    global PATH_OUTPUT
    PATH_OUTPUT = path

def getTkTkToutputPath() -> Path:
    out = PATH_OUTPUT / "tktkt"
    out.mkdir(parents=True, exist_ok=True)
    return out
