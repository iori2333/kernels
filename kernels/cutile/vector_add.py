import cuda.tile as ct
import torch


@ct.kernel
def vector_add(
    a: ct.Array,
    b: ct.Array,
    result: ct.Array,
    tile_size: ct.Constant[int] = 16,
):
    block_id = ct.bid(0)

    a_tile = ct.load(a, index=(block_id,), shape=(tile_size,))
    b_tile = ct.load(b, index=(block_id,), shape=(tile_size,))
    r_tile = a_tile + b_tile

    ct.store(result, index=(block_id,), tile=r_tile)


def launch_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "input tensors must have same shape"
    assert a.device == b.device, "input tensors must have same device"
    assert a.is_cuda, "input tensors must be CUDA"

    tile_size = 16
    r = torch.zeros_like(a)
    grid = (int(ct.cdiv(a.shape[0], tile_size)), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        vector_add,
        (a, b, r, tile_size),
    )
    return r
