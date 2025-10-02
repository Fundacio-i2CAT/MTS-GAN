"""Training Code"""
import logging
from copy import deepcopy
import torch.nn as nn
import torch
from torch.optim import Optimizer, lr_scheduler
from torch.utils import data
from tqdm import tqdm
from utils.losses import compute_d_loss, compute_g_loss
import utils.utils as ut


logger = logging.getLogger("DataLogger")


def train_discriminator(gen_net: nn.Module, dis_net: nn.Module, dis_optimizer, dis_scheduler,
                        real_ts, config: dict, ts, device, iter_idx, global_step, num_batches):
    """Discriminator Training"""
    dis_config = config["trainer"]["loss"]["d_loss"]
    z = torch.randn(ts.shape[0], config["network"]["generator"]["z_dim"], device=device)
    # Generate fake ts with generator
    fake_ts = gen_net(z).detach()
    # Pass through the discriminator network both real and fake ts
    real_validity = dis_net(real_ts)
    fake_validity = dis_net(fake_ts)

    d_loss = compute_d_loss(real_ts, fake_ts, real_validity, fake_validity,
                            dis_config, dis_net, device)

    dis_accumulated_times = config["trainer"]["d_accumulated_times"]
    # Accumulate gradients and update them each accumulated_times
    d_loss = d_loss / float(dis_accumulated_times)
    d_loss.backward()
    if (iter_idx + 1) % dis_accumulated_times == 0:
        torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
        dis_optimizer.step()
        dis_scheduler.step()
        dis_optimizer.zero_grad()

    ut.mlflow_log_metric('d_loss', d_loss.item(),
                         global_step * num_batches + iter_idx, config)
    ut.mlflow_log_metric('d_lr', dis_scheduler.get_last_lr()[0],
                         global_step * num_batches + iter_idx, config)

    return d_loss.item()


def train_generator(gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, gen_scheduler,
                    config: dict, ts, device, iter_idx, global_step, num_batches,
                    gen_avg_param):
    """Generator Training"""
    gen_config = config["trainer"]["loss"]["g_loss"]
    n_critic = config["trainer"]["n_critic"]
    dis_accumulated_times = config["trainer"]["d_accumulated_times"]
    gen_accumulated_times = config["trainer"]["g_accumulated_times"]
    # Only train generator if n_critic and accumulated times for the discrimiator
    # The discriminator needs to be more updated so the gradients are better for the generator

    if global_step % (n_critic * dis_accumulated_times) == 0:
        for _ in range(gen_accumulated_times):
            z = torch.randn(ts.shape[0], config["network"]["generator"]["z_dim"], device=device)
            fake_ts = gen_net(z)
            fake_validity = dis_net(fake_ts)
            g_loss = compute_g_loss(fake_ts, fake_validity, z, gen_config, dis_net, device)
            g_loss = g_loss / float(gen_accumulated_times)
            g_loss.backward()

        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
        gen_optimizer.step()
        gen_scheduler.step()
        gen_optimizer.zero_grad()

        ut.mlflow_log_metric('g_loss', g_loss.item(),
                             global_step * num_batches + iter_idx, config)
        ut.mlflow_log_metric('g_lr', gen_scheduler.get_last_lr()[0],
                             global_step * num_batches + iter_idx, config)

        # moving average weight
        ema_beta = config['network']['generator']['ema']

        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            cpu_p = deepcopy(p)
            avg_p.mul_(ema_beta).add_(cpu_p.cpu().data, alpha=1. - ema_beta)
            del cpu_p

        return g_loss.item()
    return float('nan')


def train(gen_net: nn.Module, dis_net: nn.Module, train_loader: data.DataLoader,
          gen_optimizer: Optimizer, dis_optimizer: Optimizer,
          gen_scheduler: lr_scheduler._LRScheduler, dis_scheduler: lr_scheduler._LRScheduler,
          config: dict, tracker: dict, device,
          gen_avg_param):
    """Main train loop"""
    gen_net.train()
    dis_net.train()
    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    global_step = tracker["train_global_step"]
    d_losses = []
    g_losses = []
    progress_bar = tqdm(train_loader, dynamic_ncols=True)

    for iter_idx, ts in enumerate(progress_bar):
        real_ts = ts.to(device, dtype=torch.float, non_blocking=True)

        d_loss = train_discriminator(gen_net, dis_net, dis_optimizer, dis_scheduler, real_ts,
                                     config, ts, device, iter_idx, global_step, len(train_loader))

        g_loss = train_generator(gen_net, dis_net, gen_optimizer, gen_scheduler,
                                 config, ts, device, iter_idx, global_step, len(train_loader),
                                 gen_avg_param)

        tracker["train_global_step"] = global_step + 1

        progress_bar.set_postfix({
            "Epoch": f"{tracker['curr_epoch']}/{tracker['end_epoch']}",
            "Batch": f"{iter_idx % len(train_loader)}/{len(train_loader)}",
            "D Loss": f"{d_loss:.6f}",
            "G Loss": f"{g_loss:.6f}"
        })

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if iter_idx == len(train_loader) - 1:
            if sum(g_losses) == sum(g_losses):  # Check if g_loss is nan
                ut.mlflow_log_metric('g_loss_avg', sum(g_losses) / len(g_losses), tracker["curr_epoch"], config)
            ut.mlflow_log_metric('d_loss_avg', sum(d_losses) / len(d_losses), tracker["curr_epoch"], config)
