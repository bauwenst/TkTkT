import time
from typing import Iterable

def dprint(d: dict):
    for e in d.items():
        print(e)


def lprint(l: Iterable):
    for e in l:
        print(e)


def kprint(d: dict, indent=0):
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


def iprint(integer: int, sep=" "):
    """
    Print an integer with a custom thousands separator.
    """
    print(intsep(integer, sep))


def intsep(integer: int, sep=" "):
    return f"{integer:,}".replace(",", sep)


def logger(msg: str):
    print("[" + time.strftime('%H:%M:%S') + "]", msg)


def warn(*msgs):
    print("[WARNING]", *msgs)


class doPrint:

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        if self.verbose:
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