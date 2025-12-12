from time import perf_counter_ns

import torch
from torch.nn.functional import softmax

from kernels.cutile import launch_softmax as cutile_softmax
from kernels.triton import launch_softmax as triton_softmax
from kernels.utils import bench_fn


def benchmark_softmax(
    launcher,
    b: int,
    s: int,
    dtype: torch.dtype,
    device: torch.device,
):
    input = torch.rand((b, s), device=device).to(dtype)
    output = torch.empty_like(input)

    def do_test():
        tic_ns = perf_counter_ns()
        launcher(input, output)
        torch.cuda.synchronize()
        toc_ns = perf_counter_ns()

        return (toc_ns - tic_ns) / 1_000_000

    return do_test


def bench():
    b, s = 1024, 2048
    device = torch.device("cuda:0")

    bench_fn(
        "cutile_softmax_bf16",
        50000,
        500,
        benchmark_softmax(
            cutile_softmax,
            b,
            s,
            dtype=torch.bfloat16,
            device=device,
        ),
    )

    bench_fn(
        "triton_softmax_bf16",
        50000,
        500,
        benchmark_softmax(
            triton_softmax,
            b,
            s,
            dtype=torch.bfloat16,
            device=device,
        ),
    )

    bench_fn(
        "cutile_softmax_fp8",
        50000,
        500,
        benchmark_softmax(
            cutile_softmax,
            b,
            s,
            dtype=torch.float8_e4m3fn,  # float8_e4m3 support seems broken
            device=device,
        ),
    )

    bench_fn(
        "triton_softmax_fp8",
        50000,
        500,
        benchmark_softmax(
            triton_softmax,
            b,
            s,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
    )


def test():
    b, s = 4096, 144
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    input = torch.rand((b, s), device=device).to(dtype)

    cutile_o = cutile_softmax(input)
    trtion_o = triton_softmax(input)
    expected = softmax(input, dim=1)

    torch.testing.assert_close(cutile_o, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(trtion_o, expected, atol=1e-3, rtol=1e-3)

    print("softmax kernel passed")


if __name__ == "__main__":
    test()
    bench()
