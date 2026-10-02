import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    ms: float = 0.0


@contextmanager
def timed():
    """`with timed() as t: ...` then read `t.ms`."""
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.ms = (time.perf_counter() - start) * 1000
