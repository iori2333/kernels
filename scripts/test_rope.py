from time import perf_counter_ns

import torch

from kernels.cutile import launch_apply_rope as cutile_apply_rope
from kernels.triton import launch_apply_rope as triton_apply_rope
from kernels.utils import bench_fn


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def torch_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def benchmark_apply_rope(
    launcher,
    b: int,
    s: int,
    hq: int,
    hk: int,
    d: int,
    dtype: torch.dtype,
    device: torch.device,
):
    q = torch.randn((b, s, hq, d), device=device).to(dtype)
    k = torch.randn((b, s, hk, d), device=device).to(dtype)
    cos = torch.randn((b, s, d), device=device).to(dtype)
    sin = torch.randn((b, s, d), device=device).to(dtype)
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    def do_test():
        tic_ns = perf_counter_ns()
        launcher(q, k, cos, sin, out_q, out_k)
        torch.cuda.synchronize()
        toc_ns = perf_counter_ns()

        return (toc_ns - tic_ns) / 1_000_000

    return do_test


def bench():
    b, s, h, d, groups = 4, 256, 8, 128, 2
    device = torch.device("cuda:0")

    bench_fn(
        "cutile_apply_rope_fp32",
        50000,
        500,
        benchmark_apply_rope(
            cutile_apply_rope,
            b,
            s,
            h,
            h // groups,
            d,
            torch.float32,
            device,
        ),
    )

    bench_fn(
        "triton_apply_rope_fp32",
        50000,
        500,
        benchmark_apply_rope(
            triton_apply_rope,
            b,
            s,
            h,
            h // groups,
            d,
            torch.float32,
            device,
        ),
    )

    bench_fn(
        "cutile_apply_rope_bf16",
        50000,
        500,
        benchmark_apply_rope(
            cutile_apply_rope,
            b,
            s,
            h,
            h // groups,
            d,
            torch.bfloat16,
            device,
        ),
    )

    bench_fn(
        "triton_apply_rope_bf16",
        50000,
        500,
        benchmark_apply_rope(
            triton_apply_rope,
            b,
            s,
            h,
            h // groups,
            d,
            torch.bfloat16,
            device,
        ),
    )


def test():
    b, s, h, d, groups = 4, 256, 8, 128, 2
    dtype = torch.float32
    device = torch.device("cuda:0")

    q = torch.randn((b, s, h, d), device=device).to(dtype)
    k = torch.randn((b, s, h // groups, d), device=device).to(dtype)
    cos = torch.randn((b, s, d), device=device).to(dtype)
    sin = torch.randn((b, s, d), device=device).to(dtype)

    cutile_oq, cutile_ok = cutile_apply_rope(q, k, cos, sin)
    triton_oq, triton_ok = triton_apply_rope(q, k, cos, sin)
    expected_q, expected_k = torch_apply_rope(q, k, cos, sin)

    torch.testing.assert_close(cutile_oq, expected_q)
    torch.testing.assert_close(triton_oq, expected_q)
    torch.testing.assert_close(cutile_ok, expected_k)
    torch.testing.assert_close(triton_ok, expected_k)

    print("apply_rope kernel passed")


if __name__ == "__main__":
    test()
    bench()
