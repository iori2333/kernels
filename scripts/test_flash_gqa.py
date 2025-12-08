from time import perf_counter_ns

import torch
from torch.nn.functional import scaled_dot_product_attention

from kernels.cutile import launch_flash_sdpa
from kernels.utils import bench_fn


def benchmark_flash_sdpa(
    launcher,
    bs: int,
    h: int,
    s: int,
    s_kv: int,
    d: int,
    groups: int,
    dtype: torch.dtype,
    device: torch.device,
):
    h_kv = h // groups

    q = torch.randn((bs, h, s, d), device=device).to(dtype)
    k = torch.randn((bs, h_kv, s_kv, d), device=device).to(dtype)
    v = torch.randn((bs, h_kv, s_kv, d), device=device).to(dtype)

    def do_test():
        tic_ns = perf_counter_ns()
        r = launcher(q, k, v, enable_gqa=True)
        torch.cuda.synchronize()
        toc_ns = perf_counter_ns()

        return (toc_ns - tic_ns) / 1_000_000

    return do_test


def test():
    bs, h, s, s_kv, d, groups = 4, 8, 128, 128, 256, 1
    h_kv = h // groups

    dtype = torch.bfloat16
    device = torch.device("cuda:0")

    q = torch.randn((bs, h, s, d), device=device).to(dtype)
    k = torch.randn((bs, h_kv, s_kv, d), device=device).to(dtype)
    v = torch.randn((bs, h_kv, s_kv, d), device=device).to(dtype)

    expected = scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        enable_gqa=True,
    )

    o = launch_flash_sdpa(q, k, v, br=32, bc=32)

    torch.testing.assert_close(o, expected)

    print("fmha test passed")


def bench():
    bs, h, s, s_kv, d, groups = 4, 8, 128, 128, 256, 1
    device = torch.device("cuda:0")

    bench_fn(
        "torch_flash_sdpa_bf16",
        10000,
        100,
        benchmark_flash_sdpa(
            scaled_dot_product_attention,
            bs,
            h,
            s,
            s_kv,
            d,
            groups,
            dtype=torch.bfloat16,
            device=device,
        ),
    )

    bench_fn(
        "cutile_flash_sdpa_bf16",
        10000,
        100,
        benchmark_flash_sdpa(
            launch_flash_sdpa,
            bs,
            h,
            s,
            s_kv,
            d,
            groups,
            dtype=torch.bfloat16,
            device=device,
        ),
    )


if __name__ == "__main__":
    bench()
