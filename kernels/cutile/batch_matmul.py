import cuda.tile as ct
import torch


@ct.kernel
def batch_matmul(
    a: ct.Array,
    b: ct.Array,
    result: ct.Array,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
):
    batch_id = ct.bid(0)
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)

    acc = ct.zeros((tm, tn), dtype=ct.float32)
    k_tiles = ct.cdiv(a.shape[2], tk)
    for k in range(k_tiles):  # type: ignore
        a_tile = ct.load(
            a,
            index=(batch_id, bid_m, k),
            shape=(1, tm, tk),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((tm, tk))
        b_tile = ct.load(
            b,
            index=(batch_id, k, bid_n),
            shape=(1, tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((tk, tn))
        acc = ct.mma(a_tile, b_tile, acc)

    acc = acc.astype(result.dtype).reshape((1, tm, tn))
    ct.store(result, index=(batch_id, bid_m, bid_n), tile=acc)


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

    result = torch.empty((bs, m, n), dtype=result_dtype, device=a.device)
    tm, tn, tk = 128, 256, 64
    grid = (bs, int(ct.cdiv(m, tm)), int(ct.cdiv(n, tn)))

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        batch_matmul,
        (a, b, result, tm, tn, tk),
    )
    return result
