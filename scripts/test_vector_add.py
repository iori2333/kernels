import torch

from kernels.triton import launch_vector_add


def test():
    vector_size = 2**12
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    a = torch.rand(vector_size, device=device).to(dtype)
    b = torch.rand(vector_size, device=device).to(dtype)
    c = launch_vector_add(a, b)

    expected = a + b
    torch.testing.assert_close(c, expected)

    print("vector_add kernel passed")


if __name__ == "__main__":
    test()
