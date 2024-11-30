"""
File that is supposed to be imported by all test files. Does output setup like any other client of the package would.
"""
from fiject import setFijectOutputFolder
from tktkt.paths import setTkTkToutputRoot, PATH_ROOT

PATH_OUT = PATH_ROOT / "data" / "out"
setTkTkToutputRoot(PATH_OUT)
setFijectOutputFolder(PATH_OUT)
