"""Optimizer"""
import logging
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, ConstantLR, LinearLR
from torch.optim import Optimizer, lr_scheduler

logger = logging.getLogger("DataLogger")


def get_optimizer(config: dict, lr: float, parameters) -> Optimizer:
    """Optimizer"""
    optimizer_name = config["trainer"]["optimizer"]["optimizer_name"]
    params = filter(lambda p: p.requires_grad, parameters)
    config_opt = config["trainer"]["optimizer"]

    if optimizer_name == "adam":
        betas = (config_opt["adam"].get("beta1", 0.9), config_opt["adam"].get("beta2", 0.999))
        logger.info("Optimizer used %s, lr: %s, beta1: %s, beta2: %s",
                    optimizer_name, lr, betas[0], betas[1])
        return Adam(params, lr=lr, betas=betas)

    elif optimizer_name == "adamw":
        weight_decay = config["adamw"].get("weight_decay", 0.01)
        logger.info("Optimizer used %s, lr: %s, weight_decay: %s",
                    optimizer_name, lr, weight_decay)
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    else:
        logger.error("Unsupported optimizer: %s", optimizer_name)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer: Optimizer, config: dict,
                  total_iters: int = 0) -> lr_scheduler._LRScheduler:
    """Scheduler"""
    scheduler_name = config["scheduler_name"]

    if scheduler_name == "linear":
        end_factor = config['linear'].get('end_factor', 0.1)
        start_factor = config['linear'].get('start_factor', 1)
        scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor,
                             total_iters=total_iters)
        logging.info("Using LinearLR with start_factor=%s, total_iters=%s, inital_lr=%s",
                     start_factor, total_iters, optimizer.defaults['lr'])
    elif scheduler_name == "exponential":
        gamma = config[scheduler_name].get("gamma", 0.999)
        scheduler = ExponentialLR(optimizer, gamma)
        logger.info("Using ExponentialLR with gamma=%s", gamma)

    elif scheduler_name == "constant":
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
        logger.info("Using ConstantLR (no decay, factor=1.0)")
    else:
        logger.error("Unsupported scheduler: %s", scheduler_name)

        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler
