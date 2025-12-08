from time import perf_counter_ns

import torch

from kernels.cutile import launch_batch_matmul as cutile_batch_matmul
from kernels.triton import launch_batch_matmul as triton_batch_matmul
from kernels.utils import bench_fn


def torch_matmul_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    result_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    bs, m, k = a.shape
    _, _, n = b.shape
    scale_a = torch.tensor(1.0, device=a.device, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=b.device, dtype=torch.float32)

    if result_dtype is None:
        result_dtype = a.dtype

    c = torch.empty((bs, m, n), device=a.device, dtype=result_dtype)
    for i in range(bs):
        # Only multiplication of row-major and column-major matrices is supported by cuBLASLt
        # So we need to transpose B to column-major view
        a_i = a[i, :, :]
        b_i = b[i, :, :].transpose(-2, -1).contiguous().transpose(-2, -1)
        c[i, :, :] = torch._scaled_mm(
            a_i,
            b_i,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )
    return c


def test():
    b, m, n, k = 16, 512, 256, 1024
    dtype = torch.float8_e4m3fn
    device = torch.device("cuda:0")

    a = torch.randn((b, m, k), device=device).to(dtype)
    b = torch.randn((b, k, n), device=device).to(dtype)

    if dtype is torch.float8_e4m3fn:
        result_dtype = torch.bfloat16  # use bfloat16 to avoid overflow
        expected = torch_matmul_fp8(a, b, result_dtype)
    else:
        result_dtype = None
        expected = torch.matmul(a, b)

    r_cutile = cutile_batch_matmul(a, b, result_dtype)
    r_triton = triton_batch_matmul(a, b, result_dtype)

    torch.testing.assert_close(r_cutile, expected)
    torch.testing.assert_close(r_triton, expected)

    print("bmm test passed")


def benchmark_bmm(
    launcher,
    bs: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    result_dtype: torch.dtype | None = None,
):
    device = torch.device("cuda:0")

    a = torch.randn((bs, m, k), device=device).to(dtype)
    b = torch.randn((bs, k, n), device=device).to(dtype)

    def do_test():
        tic_ns = perf_counter_ns()
        if result_dtype is None:
            r = launcher(a, b)
        else:
            r = launcher(a, b, result_dtype)
        torch.cuda.synchronize()
        toc_ns = perf_counter_ns()

        return (toc_ns - tic_ns) / 1_000_000

    return do_test


def bench():
    bs, m, n, k = 16, 512, 256, 1024
    dtype = torch.float8_e4m3fn

    bench_fn(
        "torch_bmm_fp8",
        10000,
        100,
        benchmark_bmm(
            torch_matmul_fp8,
            bs,
            m,
            n,
            k,
            torch.float8_e4m3fn,
            torch.bfloat16,
        ),
    )

    bench_fn(
        "triton_bmm_fp8",
        10000,
        100,
        benchmark_bmm(
            triton_batch_matmul,
            bs,
            m,
            n,
            k,
            torch.float8_e4m3fn,
            torch.bfloat16,
        ),
    )

    bench_fn(
        "cutile_bmm_fp8",
        10000,
        100,
        benchmark_bmm(
            cutile_batch_matmul,
            bs,
            m,
            n,
            k,
            torch.float8_e4m3fn,
            torch.bfloat16,
        ),
    )

    bench_fn(
        "torch_bmm_bf16",
        10000,
        100,
        benchmark_bmm(
            torch.matmul,
            bs,
            m,
            n,
            k,
            torch.bfloat16,
        ),
    )

    bench_fn(
        "triton_bmm_bf16",
        10000,
        100,
        benchmark_bmm(
            triton_batch_matmul,
            bs,
            m,
            n,
            k,
            torch.bfloat16,
        ),
    )

    bench_fn(
        "cutile_bmm_bf16",
        10000,
        100,
        benchmark_bmm(cutile_batch_matmul, bs, m, n, k, torch.bfloat16),
    )


if __name__ == "__main__":
    test()
    bench()
