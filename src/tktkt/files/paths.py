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


def relativePath(path1: Path, path2: Path):
    """
    How do I get from path1 to path2?
    Note: definitely won't work for all path pairs. Just works here.
    """
    result = ""
    for i in range(min(len(path1.parts), len(path2.parts))):
        if path1.parts[i] != path2.parts[i]:
            result += "../"*len(path1.parts[i:])
            result += "/".join(path2.parts[i:])
            break
    return Path(result)


def relativeToCwd(absolute_path: Path) -> Path:
    return relativePath(Path(os.getcwd()), absolute_path)


def from_pretrained_absolutePath(cls, absolute_path: Path):
    """
    For some reason, HuggingFace doesn't accept absolute paths for loading models. This is stupid.
    This function fixes that.
    """
    return cls.from_pretrained(relativeToCwd(absolute_path).as_posix())
