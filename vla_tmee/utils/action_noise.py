import torch


def apply_action_noise(action, noise_type: str = "cauchy"):
    """
    Apply stochastic perturbations in action space to evaluate robustness.

    Args:
        action (torch.Tensor):
            Action tensor of shape (B, T, D), where
            B is the batch size,
            T is the temporal horizon (or chunk length),
            and D = 7 * chunk_size denotes the flattened action dimension.
        noise_type (str):
            Type of noise to apply. Supported options are:
            - "cauchy": heavy-tailed Cauchy noise
            - "impulse": sparse impulse noise
            - "gaussian": zero-mean Gaussian noise

    Returns:
        torch.Tensor:
            Noisy action tensor of the same shape as the input, i.e., (B, T, D),
            with values clipped to the valid action range [-1, 1].
    """
    device = action.device
    dtype = action.dtype
    B, T, D = action.shape

    # Flatten temporal dimension for vectorized noise injection
    action_flat = action.reshape(-1, D)
    N = action_flat.shape[0]

    noise = torch.zeros_like(action_flat)

    if noise_type == "cauchy":
        # Heavy-tailed noise to simulate outliers and non-Gaussian disturbances
        gamma = 0.02
        # Inverse transform sampling for the Cauchy distribution
        eps = torch.rand((N, D), device=device, dtype=dtype)
        # Clamp to avoid numerical overflow under BF16 / FP16
        eps = torch.clamp(eps, 1e-3, 1.0 - 1e-3)
        noise = gamma * torch.tan(torch.pi * (eps - 0.5))

    elif noise_type == "impulse":
        # Sparse, high-magnitude impulse noise
        prob = 0.05
        mask = torch.rand((N, D), device=device) < prob
        impulse_values = torch.rand(
            int(mask.sum()), device=device, dtype=dtype
        ) - 0.5
        noise[mask] = impulse_values

    elif noise_type == "gaussian":
        # Standard Gaussian noise
        std = 0.02
        noise = torch.randn((N, D), device=device, dtype=dtype) * std

    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # Apply noise and clip to the valid action range
    noisy_action_flat = torch.clamp(action_flat + noise, -1.0, 1.0)

    # Reshape back to (B, T, D)
    return noisy_action_flat.view(B, T, D)