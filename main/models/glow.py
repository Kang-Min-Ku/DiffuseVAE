import torch
import torch.nn as nn
import normflows as nf


class Glow(nn.Module):
    """
    Simplified Glow model with forward_kld for loss computation and inverse mapping.
    """
    def __init__(self, width, depth, n_levels, input_dims, use_squeeze=True):
        """
        Initialize the Glow model.

        Args:
            width (int): Hidden channels width.
            depth (int): Number of Glow blocks per level.
            n_levels (int): Number of multiscale levels.
            input_dims (tuple): Tuple of (channels, height, width).
            use_squeeze (bool): Whether to apply squeezing operations.
        """
        super().__init__()
        channels, height, width = input_dims

        # Define the Glow structure (multiscale flow)
        flows = []
        for level in range(n_levels):
            # Add GlowBlocks
            flows += [
                nf.flows.GlowBlock(channels, width, scale=True)
                for _ in range(depth)
            ]
            # Add squeeze operation if enabled and dimensions allow
            if use_squeeze and height > 1 and width > 1:
                flows += [nf.flows.Squeeze()]
                channels *= 4
                height //= 2
                width //= 2
            elif not use_squeeze:
                # Exit loop if squeezing disabled
                break

        # Gaussian prior for latent variables
        latent_shape = (channels, height, width)
        self.model = nf.NormalizingFlow(nf.distributions.DiagGaussian(latent_shape), flows)

        # Store dimensions for inverse method
        self.channels = channels
        self.height = height
        self.width = width
        self.use_squeeze = use_squeeze

    def forward_kld(self, x):
        """
        Computes forward KLD loss using normflows' built-in method.
        Args:
            x (torch.Tensor): Input data tensor.
        Returns:
            torch.Tensor: Forward KLD loss.
        """
        return self.model.forward_kld(x)

    def inverse(self, z):
        """
        Inverse pass through Glow model to map latents back to input space.
        Handles multiple outputs robustly.
        """
        # Ensure z is a tensor (not tuple)
        if isinstance(z, tuple):
            z = z[0]

        # Inverse flow to reconstruct data
        result = self.model.inverse(z)

        # Unpack result safely
        if isinstance(result, tuple):
            x = result[0]  # Reconstructed input
        else:
            x = result  # If only one output is returned

        return x, None