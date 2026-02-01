import torch


def simple_mee_loss(y_pred, y_true, sigma: float = 0.5, eps: float = 1e-8):
    """
    Standard Minimum Error Entropy (MEE) loss with a Gaussian kernel.

    Args:
        y_pred, y_true: Tensors of shape (B, T, D)
        sigma: Kernel bandwidth
        eps: Numerical stability constant
    """
    e = (y_pred - y_true).view(-1, y_pred.shape[-1])  # [N, D]

    diff = e.unsqueeze(1) - e.unsqueeze(0)            # [N, N, D]
    dist_sq = (diff ** 2).sum(dim=-1)                 # [N, N]
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))   # K_ij

    loss = -torch.log(kernel.mean() + eps)
    return loss


def adaptive_mee_loss_chunk(y_pred, y_true, sigma: float = 0.5, eps: float = 1e-8):
    """
    Chunk-level adaptive MEE with sample-wise weighting.
    """
    e = (y_pred - y_true).view(-1, y_pred.shape[-1])  # [N, D]
    n = e.size(0)

    diff = e.unsqueeze(1) - e.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))

    sigma_w = torch.sqrt(torch.tensor(n / 1000.0, dtype=e.dtype, device=e.device))
    distance = torch.norm(e, dim=1)
    w = torch.exp(-distance ** 2 / (2 * sigma_w ** 2))
    w = w / (w.sum(dim=0, keepdim=True) + eps)

    # Renyi quadratic entropy estimator
    loss = -torch.log((w * kernel).sum() / (n ** 2) + eps)
    return loss


def adaptive_mee_loss_element(
    y_pred,
    y_true,
    sigma: float = 0.5,
    sigma_w: float = 0.5,
    eps: float = 1e-8,
):
    """
    Element-level adaptive MEE with pairwise reweighting.
    """
    e = (y_pred - y_true).view(-1, y_pred.shape[-1])  # [N, D]
    n = e.size(0)

    diff = e.unsqueeze(1) - e.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))

    distance = torch.norm(e, dim=1)
    # sigma_w = torch.sqrt(torch.tensor(n / 1000.0, dtype=e.dtype, device=e.device))    # if adaptive
    w = torch.exp(-distance ** 2 / (2 * sigma_w ** 2))
    w = w / (w.sum() + eps)

    W = w.view(-1, 1) * w.view(1, -1)
    V_w = (W * kernel).sum()

    loss = -torch.log(V_w + eps)
    return loss
