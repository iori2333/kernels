import math

import cuda.tile as ct
import torch

from .utils import next_power_of_2


@ct.kernel
def softmax(
    input: ct.Array,  # [b, s]
    output: ct.Array,  # [b, s]
    b: ct.Constant[int],
    ts: ct.Constant[int],  # tc >= s, tc is power of 2 for fast loading
):
    bid_b = ct.bid(0)
    blocks = ct.num_blocks(0)

    for idx in range(bid_b, b, blocks):
        line_offset = ct.arange(ts, dtype=ct.int32)

        line = ct.gather(
            input,
            (idx, line_offset),
            padding_value=-math.inf,  # type: ignore
        ).astype(ct.float32)

        line = line - ct.max(line, axis=-1, keepdims=True)
        e_line = ct.exp(line)
        o_line = e_line / ct.sum(e_line, axis=-1, keepdims=True)
        o_line = o_line.astype(input.dtype)  # type: ignore

        ct.scatter(output, (idx, line_offset), o_line)


def launch_softmax(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    assert len(input.shape) == 2, "input shape must be [b, s]"
    assert input.is_cuda, "input tensor must be CUDA"
    if output is None:
        output = torch.empty_like(input)
    assert input.shape == output.shape, "input and output must have same shape"
    assert input.device == output.device, "input and output must have same device"

    b, s = input.shape
    grid = (min(128, b),)
    tile_size = next_power_of_2(s)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        softmax,
        (input, output, b, tile_size),
    )

    return output
