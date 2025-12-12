import cuda.tile as ct
import torch

from kernels.cutile.utils import next_power_of_2


@ct.kernel
def apply_rope(
    q: ct.Array,  # [b, s, h, 2, d // 2]
    k: ct.Array,  # [b, s, h_kv, 2, d // 2]
    cos: ct.Array,  # [b, s, 2, d // 2]
    sin: ct.Array,  # [b, s, 2, d // 2]
    out_q: ct.Array,
    out_k: ct.Array,
    thq: ct.Constant[int],
    thk: ct.Constant[int],
    td: ct.Constant[int],
):
    bid_b = ct.bid(0)
    bid_s = ct.bid(1)

    cos_tile_1 = ct.load(
        cos,
        index=(bid_b, bid_s, 0, 0),
        shape=(1, 1, 1, td),
    ).reshape((1, td))

    cos_tile_2 = ct.load(
        cos,
        index=(bid_b, bid_s, 1, 0),
        shape=(1, 1, 1, td),
    ).reshape((1, td))

    sin_tile_1 = ct.load(
        sin,
        index=(bid_b, bid_s, 0, 0),
        shape=(1, 1, 1, td),
    ).reshape((1, td))

    sin_tile_2 = ct.load(
        sin,
        index=(bid_b, bid_s, 1, 0),
        shape=(1, 1, 1, td),
    ).reshape((1, td))

    q_tile_1 = ct.load(q, index=(bid_b, bid_s, 0, 0, 0), shape=(1, 1, thq, 1, td))
    q_tile_1 = q_tile_1.reshape((thq, td))
    q_tile_2 = ct.load(q, index=(bid_b, bid_s, 0, 1, 0), shape=(1, 1, thq, 1, td))
    q_tile_2 = q_tile_2.reshape((thq, td))

    # [q_1, q_2] * [cos_1, cos_2] + [-q_2, q_1] * [sin_1, sin_2]
    # [q_1cos_1 - q_2sin_1, q_2cos_2 + q_1sin_2]
    q_1 = q_tile_1 * cos_tile_1 - q_tile_2 * sin_tile_1
    q_1 = q_1.reshape((1, 1, thq, 1, td)).astype(out_q.dtype)
    q_2 = q_tile_2 * cos_tile_2 + q_tile_1 * sin_tile_2
    q_2 = q_2.reshape((1, 1, thq, 1, td)).astype(out_q.dtype)

    ct.store(out_q, index=(bid_b, bid_s, 0, 0, 0), tile=q_1)
    ct.store(out_q, index=(bid_b, bid_s, 0, 1, 0), tile=q_2)

    k_tile_1 = ct.load(k, index=(bid_b, bid_s, 0, 0, 0), shape=(1, 1, thk, 1, td))
    k_tile_1 = k_tile_1.reshape((thk, td))
    k_tile_2 = ct.load(k, index=(bid_b, bid_s, 0, 1, 0), shape=(1, 1, thk, 1, td))
    k_tile_2 = k_tile_2.reshape((thk, td))

    # [k_1, k_2] * [cos_1, cos_2] + [-k_2, k_1] * [sin_1, sin_2]
    # [k_1cos_1 - k_2sin_1, k_2cos_2 + k_1sin_2]
    k_1 = k_tile_1 * cos_tile_1 - k_tile_2 * sin_tile_1
    k_1 = k_1.reshape((1, 1, thk, 1, td)).astype(out_k.dtype)
    k_2 = k_tile_2 * cos_tile_2 + k_tile_1 * sin_tile_2
    k_2 = k_2.reshape((1, 1, thk, 1, td)).astype(out_k.dtype)

    ct.store(out_k, index=(bid_b, bid_s, 0, 0, 0), tile=k_1)
    ct.store(out_k, index=(bid_b, bid_s, 0, 1, 0), tile=k_2)


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

    q = q.view((b, s, h, 2, d // 2))
    out_q = out_q.view((b, s, h, 2, d // 2))
    k = k.view((b, s, h_kv, 2, d // 2))
    out_k = out_k.view((b, s, h_kv, 2, d // 2))
    cos = cos.view(b, s, 2, d // 2)
    sin = sin.view(b, s, 2, d // 2)

    thq = next_power_of_2(h)
    thk = next_power_of_2(h_kv)
    td = next_power_of_2(d)

    grid = (b, s)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        apply_rope,
        (q, k, cos, sin, out_q, out_k, thq, thk, td),
    )

    out_q = out_q.view((b, s, h, d))
    out_k = out_k.view((b, s, h_kv, d))

    return out_q, out_k
