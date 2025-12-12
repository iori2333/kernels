def next_power_of_2(n: int) -> int:
    if n == 0:
        return 1

    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1

    return n
