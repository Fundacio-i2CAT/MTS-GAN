"""Losses"""
import logging
import torch.nn.functional as F
import torch


logger = logging.getLogger("DataLogger")


def compute_d_loss(real_ts, fake_ts, real_validity, fake_validity,
                   config: dict, dis_net, device):
    """Discriminatr loss"""
    loss_name = config["loss_name"]

    if loss_name == 'hinge':
        d_loss = F.relu(1.0 - real_validity).mean() + F.relu(1.0 + fake_validity).mean()

    elif loss_name == 'standard':
        # soft label
        real_label = torch.full_like(real_validity, 0.9, dtype=torch.float, device=device)
        fake_label = torch.full_like(fake_validity, 0.1, dtype=torch.float, device=device)
        d_loss = F.binary_cross_entropy_with_logits(real_validity, real_label) + \
            F.binary_cross_entropy_with_logits(fake_validity, fake_label)

    elif loss_name == 'lsgan':  # Least Squares GAN
        real_label = torch.ones_like(real_validity, dtype=torch.float, device=device)
        fake_label = torch.zeros_like(fake_validity, dtype=torch.float, device=device)

        d_loss = F.mse_loss(real_validity, real_label) + F.mse_loss(fake_validity, fake_label)

    elif loss_name == 'wgangp':
        phi = config["wgangp"]["phi"]
        gradient_penalty = compute_gradient_penalty(dis_net, real_ts, fake_ts, phi, device)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + \
            gradient_penalty * 10 / (phi ** 2)

    else:
        logger.error("Discriminator Loss not implemented")
        raise NotImplementedError(loss_name)

    return d_loss


def compute_g_loss(fake_ts, fake_validity, z, g_config, dis_net, device):
    """Generator loss"""
    loss_name = g_config["loss_name"]
    g_batch_size = fake_ts.shape[0]
    if loss_name == 'standard':
        real_label = torch.ones_like(fake_validity, dtype=torch.float, device=device)
        g_loss = F.binary_cross_entropy_with_logits(fake_validity, real_label)

    elif loss_name == 'lsgan':  # Least Squares GAN
        real_label = torch.ones_like(fake_validity, dtype=torch.float, device=device)
        g_loss = F.mse_loss(fake_validity, real_label)

    elif loss_name == 'wgangp':
        # No need gradient clipng
        g_loss = -torch.mean(fake_validity)

    elif loss_name == 'wgangp-mode':
        fake_ts1, fake_ts2 = fake_ts[:g_batch_size // 2], fake_ts[g_batch_size // 2:]
        z_random1, z_random2 = z[:g_batch_size // 2], z[g_batch_size // 2:]
        lz = torch.mean(torch.abs(fake_ts2 - fake_ts1)) / torch.mean(torch.abs(z_random2 - z_random1))
        eps = 1 * 1e-5
        loss_lz = 1 / (lz + eps)
        g_loss = -torch.mean(fake_validity) + g_config.get('lambda_lz', 1.0) * loss_lz

    else:
        logger.error("Generator Loss not implemented")
        raise NotImplementedError(f"Generator loss '{loss_name}' not implemented")

    return g_loss


def compute_gradient_penalty(dis_net, real_ts, fake_ts, phi, device):
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real_ts.shape[0]

    # alpha con num. dimensiones real_ts y tam. 1 en las otras
    alpha_shape = [batch_size] + [1] * (real_ts.dim() - 1)
    alpha = torch.rand(alpha_shape, device=device)

    # interpolación entre real y fake
    interpolates = alpha * real_ts + (1 - alpha) * fake_ts
    interpolates.requires_grad_(True)  

    d_interpolates = dis_net(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient_norm = gradients.reshape(batch_size, -1).norm(2, dim=1)  # L2 norm
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()  # Squared deviation from 1

    return gradient_penalty * (10 / phi ** 2)
