from contextlib import contextmanager
from timeit import default_timer


class Timer():
    def __init__(self):
        self._elapsed_time = 0.0

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, val):
        self._elapsed_time = val


@contextmanager
def cpu_timer(log=False):
    timer = Timer()
    start = default_timer()
    yield timer
    end = default_timer()
    timer.elapsed_time = end - start
    if log:
        print(f'Elapsed time: {(end - start) * 1000} ms')

