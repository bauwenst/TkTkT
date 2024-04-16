import time
from typing import Iterable, List

def dprint(d: dict):
    """
    Print (top-level) dictionary keys and values.
    """
    for k,v in d.items():
        print(k, ":", v)


def lprint(l: Iterable):
    """
    Print list elements.
    """
    for e in l:
        print(e)


def kprint(d: dict, indent=0):
    """
    Print keys of nested dictionary.
    """
    for k,v in d.items():
        if isinstance(v, dict):
            print("\t"*indent, k, ":")
            kprint(v, indent+1)
        else:
            print("\t"*indent, k, ":", "...")


def wprint(*args, **kwargs):
    """
    Print, but surrounded by two small waits.
    Useful before and after a TQDM progress bar.
    """
    time.sleep(0.05)
    print(*args, **kwargs)
    time.sleep(0.05)


def gridify(matrix: Iterable[Iterable]) -> str:
    # Render elements and get column widths
    reprs = []
    column_widths = []
    for row in matrix:
        reprs.append([])
        for i,e in enumerate(row):
            r = str(e)
            reprs[-1].append(r)
            if i >= len(column_widths):
                column_widths.append(0)
            column_widths[i] = max(column_widths[i], len(r))

    # Format
    result = ""
    for row in reprs:
        result += "["
        L = len(row)
        for i, (repr, width) in enumerate(zip(row, column_widths)):
            result += repr + (", ")*(i != L-1) + " "*(width-len(repr))
        result += "],\n"
    return result[:-2]


def transpose(matrix: Iterable[Iterable]) -> List[List]:
    new_matrix = []
    for row in matrix:
        for y,e in enumerate(row):
            if y >= len(new_matrix):
                new_matrix.append([])
            new_matrix[y].append(e)

    return new_matrix


def iprint(integer: int, sep=" "):
    """
    Print an integer with a custom thousands separator.
    """
    print(intsep(integer, sep))


def intsep(integer: int, sep=" "):
    return f"{integer:,}".replace(",", sep)


def sgnprint(number: float):
    return f"+"*(number >= 0) + number.__repr__()


def logger(msg: str):
    print("[" + time.strftime('%H:%M:%S') + "]", msg)


def warn(*msgs):
    print("[WARNING]", *msgs)


class doPrint:

    def __init__(self, verbose=True, hesitate=False):
        self.verbose = verbose
        self.wait    = hesitate

    def __call__(self, *args, **kwargs):
        if self.verbose:
            if self.wait:
                wprint(*args, **kwargs)
            else:
                print(*args, **kwargs)


class PrintTable:
    """
    Goal: display large lists of rows like  "thing1 | thing2 | thing3"  but making sure that
    the vertical bars line up over time.

    Ayyy, worked first time.
    """

    def __init__(self, default_column_size: int=0, sep="|", end="", buffer_size=1):
        self.default = default_column_size
        self.sep = sep
        self.end = end
        self.columns = []
        self.buffer = []
        self.bs = buffer_size

    def print(self, *strings):
        while len(self.columns) < len(strings):
            self.columns.append(self.default)

        for i,s in enumerate(strings):
            s = str(s)
            if i != 0:
                print(f" {self.sep} ", end="")
            print(s, end="")

            if len(s) > self.columns[i]:
                self.columns[i] = len(s)

            print(" "*(self.columns[i] - len(s)), end="")
        print(self.end, end="")
        print()

    def delayedPrint(self, *strings):
        self.buffer.extend(strings)
        while len(self.buffer) >= self.bs:
            self.print(*self.buffer[:self.bs])
            self.buffer = self.buffer[self.bs:]

    def flushBuffer(self):
        self.print(*self.buffer)
        self.buffer = []