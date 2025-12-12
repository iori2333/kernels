import torch
import triton
import triton.language as tl


@triton.jit
def apply_rope(
    q,  # [b, s, h, d]
    k,  # [b, s, h_kv, d]
    cos,  # [b, s, d]
    sin,  # [b, s, d]
    out_q,
    out_k,
    hq: tl.constexpr,
    hk: tl.constexpr,
    half_d: tl.constexpr,
    thq: tl.constexpr,
    thk: tl.constexpr,
    td: tl.constexpr,
    stride_q_b: int,
    stride_q_s: int,
    stride_q_h: int,
    stride_k_b: int,
    stride_k_s: int,
    stride_k_h: int,
    stride_oq_b: int,
    stride_oq_s: int,
    stride_oq_h: int,
    stride_ok_b: int,
    stride_ok_s: int,
    stride_ok_h: int,
    stride_cos_b: int,
    stride_cos_s: int,
    stride_sin_b: int,
    stride_sin_s: int,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    q_ptr = q + pid_b * stride_q_b + pid_s * stride_q_s
    oq_ptr = out_q + pid_b * stride_oq_b + pid_s * stride_oq_s
    k_ptr = k + pid_b * stride_k_b + pid_s * stride_k_s
    ok_ptr = out_k + pid_b * stride_ok_b + pid_s * stride_ok_s

    offset_hq = tl.arange(0, thq)[:, None]
    mask_hq = offset_hq < hq
    offset_hk = tl.arange(0, thk)[:, None]
    mask_hk = offset_hk < hk

    offset_d_1 = tl.arange(0, td)[None, :]
    mask_d_1 = offset_d_1 < half_d
    offset_d_2 = offset_d_1 + half_d
    mask_d_2 = offset_d_2 < 2 * half_d

    q_tile_1 = tl.load(
        q_ptr + offset_hq * stride_q_h + offset_d_1,
        mask=mask_d_1 & mask_hq,
    )
    q_tile_2 = tl.load(
        q_ptr + offset_hq * stride_q_h + offset_d_2,
        mask=mask_d_2 & mask_hq,
    )

    k_tile_1 = tl.load(
        k_ptr + offset_hk * stride_k_h + offset_d_1,
        mask=mask_d_1 & mask_hk,
    )
    k_tile_2 = tl.load(
        k_ptr + offset_hk * stride_k_h + offset_d_2,
        mask=mask_d_2 & mask_hk,
    )

    cos_ptr = cos + pid_b * stride_cos_b + pid_s * stride_cos_s
    sin_ptr = sin + pid_b * stride_sin_b + pid_s * stride_sin_s

    cos_tile_1 = tl.load(cos_ptr + offset_d_1, mask=mask_d_1)
    cos_tile_2 = tl.load(cos_ptr + offset_d_2, mask=mask_d_2)
    sin_tile_1 = tl.load(sin_ptr + offset_d_1, mask=mask_d_1)
    sin_tile_2 = tl.load(sin_ptr + offset_d_2, mask=mask_d_2)

    q_1 = q_tile_1 * cos_tile_1 - q_tile_2 * sin_tile_1
    q_2 = q_tile_2 * cos_tile_2 + q_tile_1 * sin_tile_2
    tl.store(
        oq_ptr + offset_hq * stride_oq_h + offset_d_1,
        q_1,
        mask=mask_d_1 & mask_hq,
    )
    tl.store(
        oq_ptr + offset_hq * stride_oq_h + offset_d_2,
        q_2,
        mask=mask_d_2 & mask_hq,
    )

    k_1 = k_tile_1 * cos_tile_1 - k_tile_2 * sin_tile_1
    k_2 = k_tile_2 * cos_tile_2 + k_tile_1 * sin_tile_2
    tl.store(
        ok_ptr + offset_hk * stride_ok_h + offset_d_1,
        k_1,
        mask=mask_d_1 & mask_hk,
    )
    tl.store(
        ok_ptr + offset_hk * stride_ok_h + offset_d_2,
        k_2,
        mask=mask_d_2 & mask_hk,
    )


def launch_apply_rope(
    q: torch.Tensor,  # [b, s, h, d]
    k: torch.Tensor,  # [b, s, h_kv, d]
    cos: torch.Tensor,  # [b, s, d]
    sin: torch.Tensor,  # [b, s, d]
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    _, _, h_kv, d = k.shape

    if out_q is None:
        out_q = torch.empty_like(q)

    if out_k is None:
        out_k = torch.empty_like(k)

    thq = triton.next_power_of_2(h)
    thk = triton.next_power_of_2(h_kv)
    td = triton.next_power_of_2(d)

    grid = (b, s)

    apply_rope[grid](
        q,
        k,
        cos,
        sin,
        out_q,
        out_k,
        h,
        h_kv,
        d // 2,
        thq,
        thk,
        td,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        out_q.stride(0),
        out_q.stride(1),
        out_q.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
    )

    return out_q, out_k
