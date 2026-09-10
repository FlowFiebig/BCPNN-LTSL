import numpy as np


def _adaptive_sigma(centers):
    """Per-centre width = local centre spacing (one-sided at the ends).

    Zero spacings (from tied/duplicate centres) are floored to the smallest
    positive spacing so the kernels never collapse to zero width.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    d = np.diff(centers)
    spacing = np.empty(n)
    if n > 2:
        spacing[1:-1] = 0.5 * (d[:-1] + d[1:])
        spacing[0], spacing[-1] = d[0], d[-1]
    elif n == 2:
        spacing[:] = d[0]
    else:
        spacing[:] = 1.0
    pos = spacing[spacing > 0]
    spacing = np.where(spacing > 0, spacing, pos.min() if pos.size else 1.0)
    return spacing


class Encoder1D:
    def __init__(self, x_min, x_max, n_centers, sigma):
        """
        x_min, x_max : range of x
        n_centers    : number of RBFs
        sigma        : width of Gaussian kernels (scalar, or per-centre vector)
        """
        self.centers = np.linspace(x_min, x_max, n_centers)
        self.sigma = sigma

    @classmethod
    def from_quantiles(cls, x, n_centers, sigma_scale=1.0):
        """Build an encoder with centres at the empirical (midpoint) quantiles of ``x``.

        Centres sit at the ``(i + 0.5) / n`` quantiles so equal mass falls in each
        bin without wasting the two outermost centres on the single most extreme
        observations. Each centre gets an adaptive width equal to ``sigma_scale``
        times the local centre spacing (uneven by construction).
        """
        x = np.asarray(x, dtype=float)
        qs = (np.arange(n_centers) + 0.5) / n_centers
        centers = np.quantile(x, qs)
        return cls.from_centers(centers, sigma=sigma_scale * _adaptive_sigma(centers))

    @classmethod
    def from_linear_winsor(cls, x, n_centers, tail_pct=0.25, sigma=None):
        """Build a linear encoder fit on the ``tail_pct``/``100-tail_pct`` percentile range.

        Raw values are still encoded without clipping; only the centre grid endpoints
        are winsorized so tail outliers do not stretch the linear layout.
        """
        x = np.asarray(x, dtype=float)
        lo, hi = np.percentile(x, [tail_pct, 100.0 - tail_pct])
        if sigma is None:
            # Scale-invariant default: kernel width = one centre spacing of the
            # winsorized range. A bare 1/n_centers is only correct when the data
            # range has width ~1 (legacy normalized-price channels); for small
            # ranges (e.g. log returns, width ~0.02) it makes every kernel wider
            # than the whole grid and all encodings collapse to near-uniform.
            sigma = (hi - lo) / n_centers
        return cls(lo, hi, n_centers, sigma)

    @classmethod
    def from_centers(cls, centers, sigma=None):
        """Build an encoder directly from an explicit centres array.

        Used to reconstruct a decoder from saved centres. ``decode`` only needs the
        centres; ``sigma`` (scalar or per-centre vector) is derived from the local
        spacing when ``None`` so the encoder can still ``encode`` if required.
        """
        obj = cls.__new__(cls)
        obj.centers = np.asarray(centers, dtype=float)
        obj.sigma = _adaptive_sigma(obj.centers) if sigma is None else sigma
        return obj

    def encode(self, x):
        """
        x : scalar or array of shape (N,)
        Returns:
            activations of shape (N, n_centers)
        """
        x = np.atleast_1d(x)
        sigma = np.asarray(self.sigma, dtype=float)  # scalar or per-centre vector
        diff = x[:, None] - self.centers[None, :]
        log_phi = -0.5 * (diff / sigma[None, :] if sigma.ndim else diff / sigma) ** 2

        # Numerically stable softmax: subtracting the per-row max log-kernel keeps the
        # normalization well-defined even when narrow central kernels would otherwise
        # underflow every column to 0 for a far-tail point (0/0 = NaN).
        log_phi -= log_phi.max(axis=1, keepdims=True)
        phi = np.exp(log_phi)
        phi /= np.sum(phi, axis=1, keepdims=True)

        return phi

    def decode(self, activations):
        """
        activations : shape (N, n_centers)
        Returns:
            reconstructed x values shape (N,)
        """
        activations = np.atleast_2d(activations)
        x_recon = activations @ self.centers
        return x_recon

def main1(x) :
    # Example range
    x_min, x_max = -2.0, 2.0
    n_centers = 20
    sigma = (x_max - x_min) / n_centers  # good default

    encoder = RBFEncoder1D(x_min, x_max, n_centers, sigma)

    # Example data
    x = np.array([-1.3, 0.2, 1.1])

    # Encode
    z = encoder.encode(x)

    # Decode
    x_reconstructed = encoder.decode(z)

    print("Original:", x)
    print("Reconstructed:", x_reconstructed)
