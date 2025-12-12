import torch
import triton
import triton.language as tl

from .utils import as_constexpr


@triton.jit
def vector_add(pa, pb, pr, n: int, tile_size: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * tile_size
    offsets = block_start + tl.arange(0, tile_size)
    mask = offsets < n

    x = tl.load(pa + offsets, mask=mask)
    y = tl.load(pb + offsets, mask=mask)
    output = x + y

    tl.store(pr + offsets, output, mask=mask)


def launch_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "input tensors must have same shape"
    assert a.device == b.device, "input tensors must have same device"

    tile_size = 16
    r = torch.zeros_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, tile_size),)

    vector_add[grid](a, b, r, n, as_constexpr(tile_size))

    return r
