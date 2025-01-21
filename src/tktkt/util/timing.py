import time
from math import sqrt


def datetimeDashed() -> str:
    return time.strftime("%F_%X").replace(":", "-")


def timeit(func):
    """
    Decorator for measuring function's running time.
    Used above a function declaration as @timeit.

    https://stackoverflow.com/a/62905867/9352077
    """
    def measure_time(*args, **kw):
        print(f"\n=== Running {func.__qualname__} ({time.strftime('%F %X')})... ===")
        start_time = time.time()
        result = func(*args, **kw)
        print(f"=== Finished running {func.__qualname__} (took {time.time() - start_time:.2f} seconds). ===")

        return result

    return measure_time


class Timer:
    """
    Small timer class which prints the time between .lap() calls.
    Uses perf_counter instead of process_time, which is good in that it incorporates I/O and sleep time, but bad
    because evil processes like Windows Updater can hoard CPU usage and slow down the program's "actual" execution time.
    """

    def __init__(self, echo_indent: int=0):
        self._s = None
        self._t = None

        self._n    = 0
        self._sum  = 0
        self._sum2 = 0
        # self.laps = []

        self._indent = echo_indent

    def start(self, echo=False):
        if echo:
            print("\t"*self._indent + f"[Started timer at {time.strftime('%Y-%m-%d %H:%M:%S')}]")
        current_time = time.perf_counter()
        self._s = current_time
        self._t = current_time

    def lap(self, echo=False):
        current_time = time.perf_counter()
        ### Untimed zone (here happens everything in the small time between ending the previous lap and starting the next)
        delta = current_time - self._t
        self._n    += 1
        self._sum  += delta
        self._sum2 += delta**2
        # self.laps.append(delta)
        if echo:
            print("\t"*self._indent + f"[Cycle took {round(delta,5)} seconds.]")
        ###
        self._t = time.perf_counter()
        return delta

    def soFar(self, echo=False):
        total = round(time.perf_counter() - self._s, 5)
        if echo:
            print("\t"*self._indent + f"[Total runtime of {total} seconds.]")
        return total

    def lapCount(self):
        # return len(self.laps)
        return self._n

    def totalLapTime(self):
        """
        Slightly different from soFar() in that it sums all of the printed laps together, but DOESN'T include how much
        time has passed since the last lap.
        """
        # return sum(self.laps)
        return self._sum

    def averageLapTime(self):
        return self.totalLapTime() / self.lapCount() if self.lapCount() else 0

    def stdLapTime(self):
        """Standard deviation of the lap times."""
        n = self.lapCount()
        return sqrt( (self._sum2 - n * self.averageLapTime()**2) / (n-1) )
