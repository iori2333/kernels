import math

import cuda.tile as ct
import numpy as np
import torch

INV_LOG2 = 1.0 / math.log(2)


@ct.kernel
def flash_sdpa(
    q: ct.Array,  # [b, s, h, d]
    k: ct.Array,  # [b, s_kv, h_kv, d]
    v: ct.Array,  # [b, s_kv, h_kv, d]
    o: ct.Array,  # [b, s, h, d]
    qk_scale: float,
    groups: ct.Constant[int],
    br: ct.Constant[int],
    bc: ct.Constant[int],
    h: ct.Constant[int],
    d: ct.Constant[int],
):
    bid_b_h = ct.bid(0)
    bid_b = bid_b_h // h
    bid_s = ct.bid(1)
    bid_h = bid_b_h % h
    bid_hkv = bid_h // groups

    # trick: use log2 instead of loge
    qk_scale = qk_scale * INV_LOG2

    # initialize buffers
    l_i = ct.zeros((br, 1), dtype=ct.float32)
    m_i = ct.full((br, 1), -np.inf, dtype=ct.float32)
    o_i = ct.zeros((br, d), dtype=ct.float32)

    # load q_i
    q_i = ct.load(
        q,
        index=(bid_b, bid_s, bid_h, 0),
        shape=(1, br, 1, d),
    ).reshape((br, d))

    t_c = ct.cdiv(k.shape[1], bc)
    for j in range(t_c):  # type: ignore
        # load (k_j)^T and v_j
        k_jt = ct.load(
            k,
            index=(bid_b, 0, bid_hkv, j),
            shape=(1, d, 1, bc),
            order=(0, 3, 2, 1),  # transpose here
        ).reshape((d, bc))

        v_j = ct.load(
            v,
            index=(bid_b, j, bid_hkv, 0),
            shape=(1, bc, 1, d),
        ).reshape((bc, d))

        # calculate s_ij = q_i @ (k_j)^T
        s_ij = ct.zeros((br, bc), dtype=ct.float32)
        s_ij = ct.mma(q_i, k_jt, s_ij)

        # perform online softmax
        s_ij_rowmax = ct.max(s_ij, axis=1, keepdims=True)
        m_ij = max(m_i, s_ij_rowmax * qk_scale)  # type: ignore
        p_ij = ct.exp2(s_ij * qk_scale - m_ij)  # [br, bc]

        alpha = ct.exp2(m_i - m_ij)  # [br, 1]
        l_i = l_i * alpha + ct.sum(p_ij, axis=-1, keepdims=True)

        # calculate o_i = alpha * o_i-1 + p_ij @ v_j
        o_i = o_i * alpha
        p_ij = p_ij.astype(v_j.dtype)  # type: ignore
        o_i = ct.mma(p_ij, v_j, o_i)

        # write back m_i
        m_i = m_ij

    # scale o_i
    o_i = o_i / l_i
    o_i = o_i.reshape((1, br, 1, d)).astype(o.dtype)
    ct.store(o, index=(bid_b, bid_s, bid_h, 0), tile=o_i)


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
    assert q.is_cuda, "input tensors must be CUDA"

    bs, s, h, d = q.shape
    _, s_kv, h_kv, _ = k.shape

    assert h % h_kv == 0, "q_heads must be divisible by kv_heads"

    if not enable_gqa and h != h_kv:
        raise ValueError("GQA should be enabled if q_heads != kv_heads")

    groups = h // h_kv

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(d)

    if o is None:
        o = torch.empty_like(q)

    grid = (bs * h, math.ceil(s / br))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        flash_sdpa,
        (q, k, v, o, qk_scale, groups, br, bc, h, d),
    )

    return o
