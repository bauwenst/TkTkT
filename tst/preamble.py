"""
File that is supposed to be imported by all test files. Does output setup, for example.
"""
from fiject import setFijectOutputFolder
from tktkt.files.paths import setTkTkToutputRoot, getTkTkToutputPath, PATH_ROOT

PATH_OUT = PATH_ROOT / "data" / "out"

setTkTkToutputRoot(PATH_OUT)
checkpoints_path = getTkTkToutputPath() / "checkpoints"
checkpoints_path.mkdir(parents=True, exist_ok=True)

setFijectOutputFolder(PATH_OUT)