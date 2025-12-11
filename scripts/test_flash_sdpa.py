from time import perf_counter_ns

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from kernels.cutile import launch_flash_sdpa as cutile_flash_sdpa
from kernels.triton import launch_flash_sdpa as triton_flash_sdpa
from kernels.utils import bench_fn


def benchmark_flash_sdpa(
    launcher,
    b: int,
    h: int,
    s: int,
    s_kv: int,
    d: int,
    groups: int,
    dtype: torch.dtype,
    device: torch.device,
):
    h_kv = h // groups

    q = torch.randn((b, s, h, d), device=device).to(dtype)
    o = torch.randn((b, s, h, d), device=device).to(dtype)
    k = torch.randn((b, s_kv, h_kv, d), device=device).to(dtype)
    v = torch.randn((b, s_kv, h_kv, d), device=device).to(dtype)

    def do_test():
        tic_ns = perf_counter_ns()
        r = launcher(q, k, v, o, enable_gqa=True)
        torch.cuda.synchronize()
        toc_ns = perf_counter_ns()

        return (toc_ns - tic_ns) / 1_000_000

    return do_test


def launch_torch_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    enable_gqa: bool = False,
):
    # change [b, s, h, d] to [b, h, s, d]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
        o = scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
    o = o.transpose(1, 2)
    return o


def test():
    b, s, s_kv, h, d, groups = 2, 128, 128, 8, 64, 1
    h_kv = h // groups

    dtype = torch.float16
    device = torch.device("cuda:0")

    q = torch.randn((b, s, h, d), device=device).to(dtype)
    k = torch.randn((b, s_kv, h_kv, d), device=device).to(dtype)
    v = torch.randn((b, s_kv, h_kv, d), device=device).to(dtype)

    expected = launch_torch_flash_attn(q, k, v, is_causal=False, enable_gqa=True)
    o_cutile = cutile_flash_sdpa(
        q,
        k,
        v,
        br=128,
        bc=128,
        is_causal=False,
        enable_gqa=True,
    )

    o_triton = triton_flash_sdpa(
        q,
        k,
        v,
        br=128,
        bc=128,
        is_causal=False,
        enable_gqa=True,
    )

    torch.testing.assert_close(o_cutile, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(o_triton, expected, atol=1e-3, rtol=1e-3)

    print("fmha test passed")


def bench():
    bs, h, s, s_kv, d, groups = 4, 8, 256, 256, 128, 2
    device = torch.device("cuda:0")

    bench_fn(
        "cutile_flash_sdpa_bf16",
        50000,
        500,
        benchmark_flash_sdpa(
            cutile_flash_sdpa,
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
        "trtion_flash_sdpa_bf16",
        50000,
        500,
        benchmark_flash_sdpa(
            triton_flash_sdpa,
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
        "cutile_flash_sdpa_fp8",
        50000,
        500,
        benchmark_flash_sdpa(
            cutile_flash_sdpa,
            bs,
            h,
            s,
            s_kv,
            d,
            groups,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
    )

    bench_fn(
        "trtion_flash_sdpa_fp8",
        50000,
        500,
        benchmark_flash_sdpa(
            triton_flash_sdpa,
            bs,
            h,
            s,
            s_kv,
            d,
            groups,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
    )


if __name__ == "__main__":
    test()
    bench()
