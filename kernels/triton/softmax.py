import math

import torch
import triton
import triton.language as tl

from kernels.triton.utils import get_triton_dtype


@triton.jit
def softmax(
    input,  # [b, s]
    output,  # [b, s]
    b: tl.constexpr,
    s: tl.constexpr,
    input_stride: int,
    output_stride: int,
    dtype: tl.constexpr,
    ts: tl.constexpr,  # tc >= s, tc is power of 2 for fast loading
):
    pid_b = tl.program_id(0)
    blocks = tl.num_programs(0)

    for idx in range(pid_b, b, blocks):
        line_offset = tl.arange(0, ts)
        mask = line_offset < s

        line_ptr = input + idx * input_stride
        line = tl.load(
            line_ptr + line_offset,
            mask=mask,
            other=-math.inf,
        ).cast(tl.float32)

        line = line - tl.max(line, axis=-1, keep_dims=True)
        e_line = tl.exp(line)
        o_line = e_line / tl.sum(e_line, axis=-1, keep_dims=True)
        o_line = o_line.cast(dtype)

        output_ptr = output + idx * output_stride
        tl.store(output_ptr + line_offset, o_line, mask=mask)


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
    tile_size = triton.next_power_of_2(s)

    softmax[grid](
        input,
        output,
        b,
        s,
        input.stride(0),
        output.stride(0),
        get_triton_dtype(input.dtype),
        tile_size,
    )

    return output
