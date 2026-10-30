"""Small statistics helpers shared by the CLIs and experiments."""


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return [0.0, 1.0]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0.0, c - h), min(1.0, c + h)]
