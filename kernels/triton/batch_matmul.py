import torch
import triton
import triton.language as tl

from .utils import as_constexpr, as_tuple, get_triton_dtype


@triton.jit
def batch_matmul(
    a,  # [bs, m, k]
    b,  # [bs, k, n]
    r,  # [bs, m, n]
    bs: int,
    m: int,
    k: int,
    n: int,
    strides_a: tl.tuple,
    strides_b: tl.tuple,
    strides_r: tl.tuple,
    output_dtype: tl.constexpr,
    tm: tl.constexpr,
    tn: tl.constexpr,
    tk: tl.constexpr,
):
    batch_id = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    pa = tl.make_block_ptr(
        a,
        shape=(bs, m, k),
        strides=strides_a,
        offsets=(batch_id, pid_m * tm, 0),
        block_shape=(1, tm, tk),
        order=(0, 1, 2),
    )

    pb = tl.make_block_ptr(
        b,
        shape=(bs, k, n),
        strides=strides_b,
        offsets=(batch_id, 0, pid_n * tn),
        block_shape=(1, tk, tn),
        order=(0, 1, 2),
    )

    pr = tl.make_block_ptr(
        r,
        shape=(bs, m, n),
        strides=strides_r,
        offsets=(batch_id, pid_m * tm, pid_n * tn),
        block_shape=(1, tm, tn),
        order=(0, 1, 2),
    )

    acc = tl.zeros((tm, tn), dtype=tl.float32)
    k_tiles = tl.cdiv(k, tk)
    for k in range(k_tiles):
        a_tile = tl.load(
            pa,
            padding_option="zero",
            boundary_check=(2,),
        ).reshape(tm, tk)
        b_tile = tl.load(
            pb,
            padding_option="zero",
            boundary_check=(1,),
        ).reshape(tk, tn)
        acc = tl.dot(a_tile, b_tile, acc)

        pa = tl.advance(pa, (0, 0, tk))
        pb = tl.advance(pb, (0, tk, 0))

    acc = acc.cast(output_dtype).reshape(1, tm, tn)
    tl.store(pr, acc)


def launch_batch_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    result_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert a.shape[0] == b.shape[0], "inputs must have same batch size"
    assert a.device == b.device, "input tensors must be on same device"
    assert a.is_cuda, "input tensors must be CUDA"

    if result_dtype is None:
        result_dtype = a.dtype

    bs, m, ka = a.shape
    _, kb, n = b.shape
    assert ka == kb, "input tensor dims must be compatible"

    r = torch.empty((bs, m, n), dtype=result_dtype, device=a.device)
    tm, tn, tk = 128, 256, 64
    grid = (bs, triton.cdiv(m, tm), triton.cdiv(n, tn))

    dtype = get_triton_dtype(result_dtype)
    batch_matmul[grid](
        a,
        b,
        r,
        bs,
        m,
        ka,
        n,
        as_tuple(a.stride()),
        as_tuple(b.stride()),
        as_tuple(r.stride()),
        as_constexpr(dtype),
        as_constexpr(tm),
        as_constexpr(tn),
        as_constexpr(tk),
    )

    return r
