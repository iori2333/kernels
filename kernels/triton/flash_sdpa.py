import math

import torch
import triton
import triton.language as tl

from .utils import as_constexpr, as_tuple, get_triton_dtype

INV_LOG2 = tl.constexpr(1.0 / math.log(2))


@triton.jit
def flash_sdpa(
    q,
    k,
    v,
    o,
    qk_scale: float,
    groups: tl.constexpr,
    br: tl.constexpr,
    bc: tl.constexpr,
    b: tl.constexpr,
    h: tl.constexpr,
    s: tl.constexpr,
    s_kv: tl.constexpr,
    d: tl.constexpr,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    q_strides: tl.tuple,
    k_strides: tl.tuple,
    v_strides: tl.tuple,
    o_strides: tl.tuple,
):
    pid_b_h = tl.program_id(0)
    pid_b = pid_b_h // h
    pid_s = tl.program_id(1)
    pid_h = pid_b_h % h
    pid_hkv = pid_h // groups
    h_kv = h // groups

    qk_scale = qk_scale * INV_LOG2

    pq = tl.make_block_ptr(
        q,
        shape=(b, s, h, d),
        strides=q_strides,
        offsets=(pid_b, pid_s * br, pid_h, 0),
        block_shape=(1, br, 1, d),
        order=(0, 1, 2, 3),
    )

    po = tl.make_block_ptr(
        o,
        shape=(b, s, h, d),
        strides=o_strides,
        offsets=(pid_b, pid_s * br, pid_h, 0),
        block_shape=(1, br, 1, d),
        order=(0, 1, 2, 3),
    )

    kt_strides = (k_strides[0], k_strides[3], k_strides[2], k_strides[1])
    pkt = tl.make_block_ptr(
        k,
        shape=(b, d, h_kv, s_kv),
        strides=kt_strides,
        offsets=(pid_b, 0, pid_hkv, 0),
        block_shape=(1, d, 1, bc),
        order=(0, 3, 2, 1),
    )

    pv = tl.make_block_ptr(
        v,
        shape=(b, s_kv, h_kv, d),
        strides=v_strides,
        offsets=(pid_b, 0, pid_hkv, 0),
        block_shape=(1, bc, 1, d),
        order=(0, 1, 2, 3),
    )

    l_i = tl.zeros((br, 1), dtype=tl.float32)
    m_i = tl.full((br, 1), -math.inf, dtype=tl.float32)
    o_i = tl.zeros((br, d), dtype=tl.float32)

    q_i = tl.load(pq).reshape(br, d)

    t_c = tl.cdiv(s_kv, bc)
    for j in range(t_c):
        # load (k_j)^T and v_j
        pk_jt = pkt.advance((0, 0, 0, j * bc))
        k_jt = tl.load(pk_jt, boundary_check=(3,), padding_option="zero").reshape(d, bc)

        pv_j = pv.advance((0, j * bc, 0, 0))
        v_j = tl.load(pv_j, boundary_check=(1,), padding_option="zero").reshape(bc, d)

        # calculate s_ij = q_i @ (k_j)^T
        s_ij = tl.zeros((br, bc), dtype=tl.float32)
        s_ij = tl.dot(q_i, k_jt, s_ij)

        # perform online softmax
        s_ij_rowmax = tl.max(s_ij, axis=1, keep_dims=True)
        m_ij = max(m_i, s_ij_rowmax * qk_scale)
        p_ij = tl.exp2(s_ij * qk_scale - m_ij)

        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p_ij, axis=-1, keep_dims=True)

        # calculate o_i = alpha * o_i-1 + p_ij @ v_j
        o_i = o_i * alpha
        p_ij = p_ij.cast(input_dtype)
        o_i = tl.dot(p_ij, v_j, o_i)

        # write back m_i
        m_i = m_ij

    o_i = o_i / l_i
    o_i = o_i.cast(output_dtype).reshape(1, br, 1, d)
    tl.store(po, o_i)


def launch_flash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
    qk_scale: float | None = None,
    is_causal: bool = False,
    enable_gqa: bool = False,
    br: int = 64,
    bc: int = 64,
) -> torch.Tensor:
    assert not is_causal, "is_causal not implemented yet"

    assert k.shape == v.shape, "input K and V must have same shape"
    assert k.shape[0] == q.shape[0], "inputs must have same batch size"
    assert k.shape[-1] == q.shape[-1], "inputs must same hidden dims"

    assert q.device == k.device == v.device, "input tensors must be on same device"

    b, s, h, d = q.shape
    _, s_kv, h_kv, _ = k.shape

    assert h % h_kv == 0, "q_heads must be divisible by kv_heads"

    if not enable_gqa and h != h_kv:
        raise ValueError("GQA should be enabled if q_heads != kv_heads")

    groups = h // h_kv

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(d)

    if o is None:
        o = torch.empty_like(q)

    grid = (b * h, math.ceil(s / br))
    input_dtype = get_triton_dtype(q.dtype)
    output_dtype = get_triton_dtype(o.dtype)
    flash_sdpa[grid](
        q,
        k,
        v,
        o,
        qk_scale,
        as_constexpr(groups),
        as_constexpr(br),
        as_constexpr(bc),
        as_constexpr(b),
        as_constexpr(h),
        as_constexpr(s),
        as_constexpr(s_kv),
        as_constexpr(d),
        as_constexpr(input_dtype),
        as_constexpr(output_dtype),
        as_tuple(q.stride()),
        as_tuple(k.stride()),
        as_tuple(v.stride()),
        as_tuple(o.stride()),
    )
    return o
