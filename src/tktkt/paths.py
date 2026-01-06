from pathlib import Path
import os
import re

PATH_PACKAGE = Path(__file__).resolve().parent
PATH_ROOT    = PATH_PACKAGE.parent.parent  # Can only be accessed if the package was installed in editable mode.

# Output setup: if we can detect that you're running from inside the package project, we assume you don't want CWD.
PATH_OUTPUT = Path(os.getcwd())
if PATH_OUTPUT.is_relative_to(PATH_ROOT):  # is_relative_to means "is a descendant of"
    PATH_OUTPUT = PATH_ROOT / "data" / "out"

def setTkTkToutputRoot(path: Path):
    global PATH_OUTPUT
    PATH_OUTPUT = path


def relativePath(path1: Path, path2: Path):
    """
    How do I get from path1 to path2?
    TODO: Doesn't work if you have different disks.
    """
    result = ""
    L = min(len(path1.parts), len(path2.parts))
    for i in range(L):
        if path1.parts[i] != path2.parts[i]:
            result += "../"*len(path1.parts[i:])
            result += "/".join(path2.parts[i:])
            break
    else:  # One path is a prefix of the other. Below, the slice [L:] will be empty for one or both of them.
        result += "../"*len(path1.parts[L:])
        result += "/".join(path2.parts[L:])

    return Path(result)


def relativeToCwd(absolute_path: Path) -> Path:
    return relativePath(Path(os.getcwd()), absolute_path)


def from_pretrained_absolutePath(cls, absolute_path: Path):
    """
    For some reason, HuggingFace doesn't accept absolute paths for loading models. This is stupid.
    This function fixes that.
    """
    return cls.from_pretrained(relativeToCwd(absolute_path).as_posix())


def pathSafe(string: str) -> str:
    """
    Adapted from https://stackoverflow.com/a/71199182/9352077 which is based on https://en.wikipedia.org/wiki/Filename#Reserved_characters_and_words.
    Illegal strings are prefixed by an underscore.
    """
    placeholder = "_"  # This is not an argument of the function because then the user could give an illegal placeholder.

    string = string.strip()
    string = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", placeholder, string)

    # Don't allow dots at the end
    string = string.rstrip(".")

    # If empty or part of Windows's illegal list of names, add an extra placeholder at the end.
    if not string or string.upper() in {"CON", "CONIN$", "CONOUT$", "PRN", "AUX", "CLOCK$", "NUL",
                                        "COM0", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                                        "LPT0", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}:
        string += "_"

    return string


class PathManager:

    def __init__(self, project_name: str=""):
        self.project = project_name.replace(".", "").replace("/", "") or "tktkt"

    # Define general path operations as if paths are lists:

    @staticmethod
    def append(base_path: Path, part: str) -> Path:
        full_path = base_path / pathSafe(part)
        full_path.mkdir(exist_ok=True, parents=True)
        return full_path

    @staticmethod
    def extend(base_path: Path, parts: list[str]) -> Path:
        full_path = base_path
        for part in parts:
            full_path /= pathSafe(part)
        full_path.mkdir(exist_ok=True, parents=True)
        return full_path

    @staticmethod
    def files(base_path: Path) -> list[Path]:
        try:
            _, _, filenames = next(base_path.walk())
        except AttributeError:  # Python < 3.12
            import os
            _, _, filenames = next(os.walk(base_path))
        return [base_path / filename for filename in sorted(filenames)]

    @staticmethod
    def folders(base_path: Path) -> list[Path]:
        try:
            _, subfolders, _ = next(base_path.walk())
        except AttributeError:  # Python < 3.12
            import os
            _, subfolders, _ = next(os.walk(base_path))
        return [base_path / folder for folder in sorted(subfolders)]

    # Define typical paths in any ML context:

    def pathToModels(self, *subfolders: str) -> Path:
        """
        For final tokeniser (or other) models.
        """
        return self._extendOutput(["models"] + list(subfolders))

    def pathToCheckpoints(self, *subfolders: str) -> Path:
        """
        For checkpoints of models trained with gradient descent.
        """
        return self._extendOutput(["checkpoints"] + list(subfolders))

    def pathToEvaluations(self, *subfolders: str) -> Path:
        """
        For numerical results.
        """
        return self._extendOutput(["evaluations"] + list(subfolders))

    # Methods for automatically providing arguments to append/extend:

    def _homeDirectory(self) -> Path:
        out = PATH_OUTPUT / self.project
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _extendOutput(self, parts: list[str]) -> Path:
        return PathManager.extend(self._homeDirectory(), parts)


TkTkTPaths = PathManager()
