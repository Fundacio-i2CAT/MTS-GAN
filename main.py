import argparse
from argparse import Namespace
from torch.utils import data
import mlflow
import logging
import h5py
import torch

from copy import deepcopy
import numpy as np
import core.trainer as trainer
import core.validator as validator
from data.dataLoader import IdekoDatasetLoader
from architecture import discriminator, generator
import utils.utils as ut
import utils.optimizer as opt
import utils.logger_ut as log_ut

logger = logging.getLogger("DataLogger")

log_ut.setup_logger(debug=True)


def main(args: Namespace):

    config = ut.load_yaml(args.config)
    ut.mlflow_init(config["mlflow"])
    ut.set_random_seed(config["data"]["random_seed"])
    device = ut.set_device()

    # Initialization Dataloaders
    with h5py.File(config['data']['dataset_path'], 'r') as f:
        train_data = f['train'][:]
        validation_data = f['val'][:]

    train_set = IdekoDatasetLoader(train_data, norm=config['data']['normalization'])
    validation_set = IdekoDatasetLoader(validation_data, norm=config['data']['normalization'], stats=train_set.stats)
    
    def add_extra_dim(batch):
        # Transpose to (num_samples, num_channels, max_rows)
        batch = np.transpose(batch, (0, 2, 1))  
        # Reshape to (num_samples, num_channels, 1, seq_len)
        batch = batch.reshape(batch.shape[0], config["data"]["num_channels"], 1, config["data"]["seq_len"])
        batch = torch.from_numpy(batch).float()  
        return batch

    train_loader = data.DataLoader(train_set, batch_size=config["trainer"]["batch_size"], shuffle=True, collate_fn=add_extra_dim)
    validation_loader = data.DataLoader(validation_set, batch_size=config["trainer"]["batch_size"], shuffle=True, collate_fn=add_extra_dim)

    # Network initialization
    gen_net = generator.Generator(config["network"]["generator"])
    dis_net = discriminator.Discriminator(config["network"]["discriminator"])

    ut.mlflow_log_summary(gen_net, dis_net, config)
    ut.mlflow_log_artifact(args.config, None, config)
    ut.mlflow_log_params(ut.flatten_dict(config), config)

    gen_net.to(device)
    dis_net.to(device)
    
    gen_net.apply(lambda m: ut.weights_init(config["network"]["init_type"], m))
    dis_net.apply(lambda m: ut.weights_init(config["network"]["init_type"], m))

    # Optimizer and scheduler initialization
    gen_optimizer = opt.get_optimizer(config, config["trainer"]["g_lr"], gen_net.parameters())
    dis_optimizer = opt.get_optimizer(config, config["trainer"]["d_lr"], dis_net.parameters())

    gen_scheduler = opt.get_scheduler(gen_optimizer, config["trainer"]['scheduler']["g_scheduler"], total_iters=config["trainer"]["epochs"] * len(train_loader))
    dis_scheduler = opt.get_scheduler(dis_optimizer, config["trainer"]['scheduler']["d_scheduler"], total_iters=config["trainer"]["epochs"] * len(train_loader))
    
    # Exponential Moving Average 
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = ut.copy_params(avg_gen_net)
    del avg_gen_net

    end_epoch = config["trainer"]["epochs"]
    # TODO change start epoch in case it is loaded from a checkpint
    start_epoch = 0
    val_start_epoch = 0
    model_not_improved = 0
    val_metric = config["data"].get("val_score", "fid_score")
    
    if val_metric in ["fid_score", "jensen_shannon_divergence"]:
        best_val_metric = float("inf")
    elif val_metric == "cosine_similarity":
        best_val_metric = float("-inf")
    else:
        raise ValueError(f"Validation metric '{val_metric}' not supported. Choose from 'fid_score', \
                         'jensen_shannon_divergence', or 'cosine_similarity'.")
    
    logger.info(f"Validation metric used: {val_metric}")
    tracker = {"train_global_step": start_epoch,
               "validation_global_step": val_start_epoch,
               "end_epoch": end_epoch}
    
    for epoch in range(int(start_epoch), int(end_epoch)):
        tracker["curr_epoch"] = epoch
        trainer.train(gen_net, dis_net, train_loader,
                      gen_optimizer, dis_optimizer,
                      gen_scheduler, dis_scheduler, config, tracker, device,
                      gen_avg_param)
        
        # Validtion and checkpoitn saving
        if epoch % config["validator"]["val_freq"] == 0:
            computed_metrics = validator.validate(gen_net, validation_loader, config, tracker, device)
            validator.gen_plot(gen_net, epoch, config)

            avg_gen_net = deepcopy(gen_net)
            ut.load_params(avg_gen_net, gen_avg_param, args)
            fid_score = computed_metrics["average_fid_score"]
            js_div = computed_metrics["average_jensen_shannon_divergence"]
            cosine_sim = computed_metrics["average_cosine_similarity"]

            logger.info(f"Validation completed for Epoch {epoch}/{int(end_epoch)}. \n" +
                        f"FID Score: {fid_score:.4f} \n" +
                        f"Jensen-Shannon Divergence: {js_div:.4f} \n" +
                        f"Cosine Similarity: {cosine_sim:.4f} \n" +
                        f"Best validation metric: {best_val_metric:.4f}")

            if (val_metric == "cosine_similarity" and cosine_sim > best_val_metric) or \
               (val_metric != "cosine_similarity" and computed_metrics["average_" + val_metric] < best_val_metric):

                logger.info("Improvemenent on the FID score saving model")
                best_fid_score = fid_score
                ut.mlflow_save_checkpoint({
                        'epoch': epoch + 1,
                        'gen_state_dict': gen_net.state_dict(),
                        'dis_state_dict': dis_net.state_dict(),
                        'avg_gen_state_dict': avg_gen_net.state_dict(),
                        'gen_optimizer': gen_optimizer.state_dict(),
                        'dis_optimizer': dis_optimizer.state_dict(),
                        'model_config': config["network"]
                    }, config=config)
                best_fid_score = fid_score
                best_jensen_shannon_divergence = js_div
                best_cosine_alignment = cosine_sim
                best_val_metric = computed_metrics["average_"+val_metric]
                model_not_improved = 0

                if model_not_improved >= config["trainer"]["early_stopping"]:
                    print(f"Model not improved in {model_not_improved}, stoping training")
                    # break

            del avg_gen_net

    if config["mlflow"]["is_used"]:
        ut.mlflow_log_metric("best_fid_score", best_fid_score, None, config)
        ut.mlflow_log_metric("best_jensen_shannon_divergence", best_jensen_shannon_divergence, None, config)
        ut.mlflow_log_metric("best_cosine_similarity", best_cosine_alignment, None, config)
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    main(args)