"""Main script"""
import argparse
import logging
from copy import deepcopy
from argparse import Namespace
import mlflow
import h5py
from torch.utils import data
import torch
import numpy as np
import core.trainer as trainer
import core.validator as validator
from data.dataLoader import CustomDataset
from architecture.model import ModelFactory
import utils.utils as ut
import utils.optimizer as opt
import utils.logger_ut as log_ut


logger = logging.getLogger("DataLogger")

log_ut.setup_logger(debug=True)


def main(args: Namespace):
    """Main func"""
    config = ut.load_yaml(args.config)
    ut.mlflow_init(config["mlflow"])
    ut.set_random_seed(config["data"]["random_seed"])
    device = ut.set_device()

    # Initialization Dataloaders
    with h5py.File(config['data']['dataset_path'], 'r') as f:
        train_data = f['train'][:]
        validation_data = f['val'][:]

    train_data = train_data[:, :config["data"]["seq_len"], :]
    validation_data = validation_data[:, :config["data"]["seq_len"], :]

    train_set = CustomDataset(train_data, norm=config['data']['normalization'])
    validation_set = CustomDataset(validation_data, norm=config['data']['normalization'],
                                   stats=train_set.stats)

    if config["type"] == "tts-gan":
        def add_extra_dim(batch):
            # Transpose to (num_samples, num_channels, max_rows)
            batch = np.transpose(batch, (0, 2, 1))
            # Reshape to (num_samples, num_channels, 1, seq_len)
            batch = batch.reshape(batch.shape[0], config["data"]["num_channels"],
                                  1, config["data"]["seq_len"])
            batch = torch.from_numpy(batch).float()
            return batch
    else:
        def add_extra_dim(batch):

            batch = np.array(batch)
            batch = torch.from_numpy(batch).float()
            return batch

    train_loader = data.DataLoader(train_set, batch_size=config["trainer"]["batch_size"],
                                   shuffle=True, collate_fn=add_extra_dim)
    validation_loader = data.DataLoader(validation_set, batch_size=config["trainer"]["batch_size"],
                                        shuffle=True, collate_fn=add_extra_dim)

    # Network initialization
    gen_net, dis_net = ModelFactory.create_model(config)

    gen_net.to(device)
    dis_net.to(device)

    ut.mlflow_log_summary(gen_net, dis_net, config)
    ut.mlflow_log_artifact(args.config, None, config)
    ut.mlflow_log_params(ut.flatten_dict(config), config)

    gen_net.apply(lambda m: ut.weights_init(config["network"]["init_type"], m))
    dis_net.apply(lambda m: ut.weights_init(config["network"]["init_type"], m))

    # Optimizer and scheduler initialization
    gen_optimizer = opt.get_optimizer(config, config["trainer"]["g_lr"], gen_net.parameters())
    dis_optimizer = opt.get_optimizer(config, config["trainer"]["d_lr"], dis_net.parameters())

    gen_scheduler = opt.get_scheduler(gen_optimizer, config["trainer"]['scheduler']["g_scheduler"],
                                      total_iters=config["trainer"]["epochs"] * len(train_loader))
    dis_scheduler = opt.get_scheduler(dis_optimizer, config["trainer"]['scheduler']["d_scheduler"],
                                      total_iters=config["trainer"]["epochs"] * len(train_loader))

    # Exponential Moving Average
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = ut.copy_params(avg_gen_net)
    del avg_gen_net

    end_epoch = config["trainer"]["epochs"]

    start_epoch = 0
    val_start_epoch = 0
    model_not_improved = 0
    val_metric = config["data"].get("val_score", "fid_score_statistic")
    best_val_metric = ut.set_validation_metric(val_metric)
    best_fid_score_st = float("inf")
    best_jensen_shannon_divergence = float("inf")
    best_js_div_st = float("inf")
    best_cosine_alignment_st = float("-inf")
    best_cosine_alignment = float("-inf")

    logger.info("Validation metric used: %s", val_metric)
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
            computed_metrics = validator.validate(gen_net, validation_loader,
                                                  config, tracker, device)
            validator.gen_plot(gen_net, epoch, config)

            avg_gen_net = deepcopy(gen_net)
            ut.load_params(avg_gen_net, gen_avg_param, args)

            fid_score_st = computed_metrics["average_fid_score_statistic"]
            js_div_st = computed_metrics["average_jensen_shannon_divergence_statistic"]
            cosine_sim_st = computed_metrics["average_cosine_similarity_statistic"]
            js_div = computed_metrics["average_jensen_shannon_divergence"]
            cosine_sim = computed_metrics["average_cosine_similarity"]

            if (val_metric == "cosine_similarity" and cosine_sim > best_val_metric) or \
               (val_metric != "cosine_similarity" and computed_metrics["average_" + val_metric] < best_val_metric):

                logger.info("Improvement on the FID score saving model, old: %s, New best: %s",
                            best_val_metric, computed_metrics['average_' + val_metric])
                ut.mlflow_save_checkpoint({'epoch': epoch + 1,
                                           'gen_state_dict': gen_net.state_dict(),
                                           'dis_state_dict': dis_net.state_dict(),
                                           'avg_gen_state_dict': avg_gen_net.state_dict(),
                                           'gen_optimizer': gen_optimizer.state_dict(),
                                           'dis_optimizer': dis_optimizer.state_dict(),
                                           'model_config': config["network"]}, config=config)

                best_fid_score_st = fid_score_st
                best_js_div_st = js_div_st
                best_cosine_alignment_st = cosine_sim_st
                best_jensen_shannon_divergence = js_div
                best_cosine_alignment = cosine_sim
                best_val_metric = computed_metrics["average_" + val_metric]
                model_not_improved = 0

            else:
                model_not_improved += 1

                logger.info("No improvement on the validation metric. Current best: %.4f. Model not improved for %s epochs.",
                            best_val_metric, model_not_improved)

            if model_not_improved >= config["trainer"]["early_stopping"]:
                print(f"Model not improved in {model_not_improved}, stoping training")
                break

            del avg_gen_net
    ut.mlflow_save_checkpoint({'epoch': "last",
                                        'gen_state_dict': gen_net.state_dict(),
                                        'dis_state_dict': dis_net.state_dict(),
                                        'gen_optimizer': gen_optimizer.state_dict(),
                                        'dis_optimizer': dis_optimizer.state_dict(),
                                        'model_config': config["network"]}, config=config)
    if config["mlflow"]["is_used"]:
        ut.mlflow_log_metric("best_val_metric", best_val_metric, None, config)
        ut.mlflow_log_metric("best_fid_score_st", best_fid_score_st, None, config)
        ut.mlflow_log_metric("best_cosine_similarity_st", best_cosine_alignment_st, None, config)
        ut.mlflow_log_metric("best_js_div_st", best_js_div_st, None, config)
        ut.mlflow_log_metric("best_jensen_shannon_divergence", best_jensen_shannon_divergence,
                             None, config)
        ut.mlflow_log_metric("best_cosine_similarity", best_cosine_alignment, None, config)
        mlflow.end_run()

    return best_jensen_shannon_divergence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config_tts.yaml",
                        help="Path to YAML config file")
    argss = parser.parse_args()
    main(argss)
