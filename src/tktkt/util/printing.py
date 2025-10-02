import time
from typing import Iterable, List
import logging


def setLoggingLevel(level: int=logging.WARNING):
    """
    For some reason this is necessary to change the logging level.
    https://stackoverflow.com/a/73284328/9352077
    """
    logger  = logging.getLogger()
    channel = logging.StreamHandler()
    logger .setLevel(logging.INFO)
    channel.setLevel(logging.INFO)
    logger.addHandler(channel)


def dprint(d: dict, indent: int=0):
    """
    Print (top-level) dictionary keys and values.
    """
    if not hasattr(d, "items"):
        d = d.__dict__
    
    for k,v in d.items():
        if indent:
            print("\t"*indent, end="")
        print(k, ":", v)


def rprint(d: dict, indent: int=0):
    """
    Recursive dictionary printing.
    """
    print("{")
    for k,v in d.items():
        print("\t"*(indent+1), end="")
        print(k, end="")
        print(": ", end="")
        if isinstance(v, dict):
            rprint(v, indent+1)
        else:
            print(v)
    print("\t"*indent, end="")
    print("}")


def lprint(l: Iterable, indent: int=0):
    """
    Print list elements.
    """
    for e in l:
        if indent:
            print("\t"*indent, end="")
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


def iprint(integer: int, sep=" "):
    """
    Print an integer with a custom thousands separator.
    """
    print(intsep(integer, sep))


def intsep(integer: int, sep=" ") -> str:
    return f"{integer:,}".replace(",", sep)


def percent(num: int, denom: int, decimals: int=2) -> str:
    return (f"{round(100*num/denom, decimals)}" if denom != 0 else "???") + "%"


def inequality(a: float, b: float):
    if a == b:
        op = "="
    elif a > b:
        op = ">"
    else:
        op = "<"
    return f"{a} {op} {b}"


def sgnprint(number: float) -> str:
    return f"+"*(number >= 0) + number.__repr__()


def pluralise(number: int, singular: str, plural_suffix: str="s", plural: str="") -> str:
    if not plural:
        plural = singular + plural_suffix
    return f"{number} " + (singular if number == 1 else plural)  # 0 dogs, 1 dog, 2 dogs


def ordinal(number: int) -> str:
    return f"{number}{'st' if number == 1 else 'nd' if number == 2 else 'rd' if number == 3 else 'th'}"


def roundHuman(number: float, round_to: int=2, base2: bool=False, trim_zeroes: bool=True):
    """
    Round numbers up to the given amount of digits + 1, hiding the rest in a letter suffix like K, M, G, ...
    For example, 999987 becomes 999.99K when rounded to two decimals. Rounded to one decimal it becomes 1.0M.
    Adaptation of https://stackoverflow.com/a/49955617/9352077.
    If you want something even more powerful (including for values in [-1,1]), see https://stackoverflow.com/a/71833808/9352077.
    """
    if base2:
        base     = 1024
        suffixes = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi']
    else:
        base     = 1000
        suffixes = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']

    magnitude = 0
    while abs(number) >= base and magnitude < len(suffixes)-1:
        number = round(number / base, round_to)
        magnitude += 1
    return f"{number:.{round_to}f}".rstrip("0"*trim_zeroes).rstrip(".") + suffixes[magnitude]


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

    Will not work when the characters in the table have mixed widths (e.g. English and Japanese words).
    """

    def __init__(self, default_column_size: int=0, sep="|", end="", full_width_space: bool=False, buffer_size=1):
        """
        :param full_width_space: Whether to use the spacing for wider characters.
        """
        self.default = default_column_size
        self.sep = sep
        self.end = end
        self.spacer = "　" if full_width_space else " "
        self.columns = []

        self.buffer = []
        self.bs     = buffer_size

    def print(self, *strings):
        while len(self.columns) < len(strings):
            self.columns.append(self.default)

        for i,s in enumerate(strings):
            s = str(s)
            if i != 0:
                print(f"{self.spacer}{self.sep}{self.spacer}", end="")
            print(s, end="")

            if len(s) > self.columns[i]:
                self.columns[i] = len(s)

            print(self.spacer*(self.columns[i] - len(s)), end="")
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