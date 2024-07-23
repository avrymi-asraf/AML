import torch
import torch.nn.functional as F


def invariance_loss(z1, z2):
    """
    Compute the invariance loss between two sets of projections.

    Args:
    z1, z2 (torch.Tensor): Batch of projections, shape (batch_size, projection_dim)

    Returns:
    torch.Tensor: Scalar invariance loss
    """
    return F.mse_loss(z1, z2)


def variance_loss(z, gamma=1, epsilon=1e-4):
    """
    Compute the variance loss for a batch of projections.

    Args:
    z (torch.Tensor): Batch of projections, shape (batch_size, projection_dim)
    gamma (float): Target standard deviation
    epsilon (float): Small constant for numerical stability

    Returns:
    torch.Tensor: Scalar variance loss
    """
    std_z = torch.sqrt(z.var(dim=0) + epsilon)
    return torch.mean(F.relu(gamma - std_z))


def covariance_loss(z):
    """
    Compute the covariance loss for a batch of projections.

    Args:
    z (torch.Tensor): Batch of projections, shape (batch_size, projection_dim)

    Returns:
    torch.Tensor: Scalar covariance loss
    """
    N, D = z.shape
    z = z - z.mean(dim=0)
    cov_z = (z.T @ z) / (N - 1)
    diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    return (cov_z[diag_mask] ** 2).sum() / D


def vicreg_loss_detailed(z1, z2, lambda_inv=25, lambda_var=25, lambda_cov=1):
    """
    Compute the combined VICReg loss with detailed component breakdown.

    Args:
    z1, z2 (torch.Tensor): Batch of projections from two views, shape (batch_size, projection_dim)
    lambda_inv (float): Weight for invariance loss
    lambda_var (float): Weight for variance loss
    lambda_cov (float): Weight for covariance loss

    Returns:
    torch.Tensor: Scalar VICReg loss
    dict: Dictionary containing individual loss components
    """
    inv_loss = invariance_loss(z1, z2)
    var_loss = (variance_loss(z1) + variance_loss(z2)) / 2
    cov_loss = (covariance_loss(z1) + covariance_loss(z2)) / 2

    total_loss = lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss

    loss_components = {
        "inv_loss": inv_loss.item(),
        "var_loss": var_loss.item(),
        "cov_loss": cov_loss.item(),
        "loss": total_loss.item(),
    }

    return total_loss, loss_components


def vicreg_loss_performance(z1, z2, lambda_inv=25, lambda_var=25, lambda_cov=1):
    """
    Compute the combined VICReg loss optimized for performance.

    Args:
    z1, z2 (torch.Tensor): Batch of projections from two views, shape (batch_size, projection_dim)
    lambda_inv (float): Weight for invariance loss
    lambda_var (float): Weight for variance loss
    lambda_cov (float): Weight for covariance loss

    Returns:
    torch.Tensor: Scalar VICReg loss
    """
    # Invariance loss
    inv_loss = F.mse_loss(z1, z2)

    # Variance loss
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))) / 2

    # Covariance loss
    N, D = z1.shape
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)
    diag_mask = ~torch.eye(D, dtype=torch.bool, device=z1.device)
    cov_loss = ((cov_z1[diag_mask] ** 2).sum() + (cov_z2[diag_mask] ** 2).sum()) / (
        2 * D
    )

    return lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss
