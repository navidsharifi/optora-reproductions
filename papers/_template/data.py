"""Data of the experiment.

A real paper downloads its dataset here (with a checksum) or generates the
synthetic data the paper describes. Data is never committed.
"""

from __future__ import annotations

import torch


def sample_losses(
    num_samples: int,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Draw i.i.d. standard exponential losses.

    The exponential distribution is skewed and nonnegative, like many
    training losses.

    Args:
        num_samples: Number of losses.
        generator: CPU generator the draws come from.
        dtype: Floating-point type of the result.
        device: Device of the result.

    Returns:
        Tensor of shape `(num_samples,)`.
    """
    losses = torch.empty(num_samples, dtype=dtype).exponential_(generator=generator)
    return losses.to(device)
