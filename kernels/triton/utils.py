from typing import Any

import torch
import triton.language as tl

PT_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.int32: tl.int32,
    torch.int16: tl.int16,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
}


def get_triton_dtype(pt_dtype: torch.dtype):
    if pt_dtype not in PT_TO_TRITON_DTYPE:
        raise ValueError(f"Unsupported PyTorch dtype: {pt_dtype}")
    return PT_TO_TRITON_DTYPE[pt_dtype]


# make pylance happy
def as_constexpr(value: Any) -> tl.constexpr:
    return value


def as_tuple(tup: tuple[Any, ...]) -> tl.tuple:
    return tup  # type: ignore
