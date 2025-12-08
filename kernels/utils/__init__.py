from time import perf_counter_ns
from typing import Any, Callable


def bench_fn[**P](
    name: str,
    reps: int,
    warmup_reps: int,
    fn: Callable[P, float],
    *args: P.args,
    **kwargs: P.kwargs,
):
    print(f"warming up benchmark {name}...", end="\r")
    for _ in range(warmup_reps):
        fn(*args, **kwargs)

    print(f"starting benchmark {name}...", end="\r")

    tic_ms = 0.0
    for _ in range(reps):
        tic_ms += fn(*args, **kwargs)
    avg_ms = tic_ms / reps

    print(f"benchmark {name} done, {avg_ms:.4f} ms")
