import time


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

    def __init__(self):
        self.s = None
        self.t = None
        self.laps = []

    def start(self, echo=False):
        if echo:
            print(f"    [Started timer at {time.strftime('%Y-%m-%d %H:%M:%S')}]")
        current_time = time.perf_counter()
        self.s = current_time
        self.t = current_time

    def lap(self, echo=False):
        current_time = time.perf_counter()
        ### Safe zone (prints and other operations between ending the previous lap and starting the next)
        delta = round(current_time - self.t, 5)
        self.laps.append(delta)
        if echo:
            print(f"    [Cycle took {delta} seconds.]")
        ###
        self.t = time.perf_counter()
        return delta

    def soFar(self, echo=False):
        total = round(time.perf_counter() - self.s, 5)
        if echo:
            print(f"    [Total runtime of {total} seconds.]")
        return total

    def lapCount(self):
        return len(self.laps)

    def totalLapTime(self):
        """
        Slightly different from soFar() in that it sums all of the printed laps together, not including the current one.
        """
        return sum(self.laps)

    def averageLapTime(self):
        return self.totalLapTime() / self.lapCount() if self.lapCount() else 0
