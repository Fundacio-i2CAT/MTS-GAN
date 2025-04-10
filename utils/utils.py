import yaml
import os
import torch
import torch.nn as nn
import random
import numpy as np
import mlflow
import logging
import pandas as pd
from torchinfo import summary
from copy import deepcopy


logger = logging.getLogger("DataLogger")


def mlflow_save_checkpoint(states, config):
    if config["mlflow"]["is_used"]:
        temp_filename = f"{states['epoch']}_checkpoint.checkpoint"
        temp_dir = "/data/tmp_extreme_xp/"  # or any other directory where you want to store the temp file
        tmpfile_name = os.path.join(temp_dir, temp_filename)
        torch.save(states, tmpfile_name)
        mlflow.log_artifact(tmpfile_name, artifact_path="checkpoints")
        os.remove(tmpfile_name)
        logger.info("Model looged in MLFLOW")
    

def load_yaml(file_path: str) -> dict:
 
    if not os.path.exists(file_path):
        logger.error(f"Yaml file not found in {file_path}")
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
        
    except Exception as e:
        logger.error(f"Unexpected error while loading {file_path}")
        raise Exception(f"Unexpected error loading '{file_path}': {e}")


def set_random_seed(seed: int):
    if seed is not None:
        logger.info(f"Seting random seed: {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.info("Random seed not in use")


def mlflow_init(config: dict):
    # TODO if mlflow not used init loging values
    if config.get("is_used", False):
        mlflow.set_tracking_uri(config["URI"])
        
        experiment = mlflow.get_experiment_by_name(config["experiment_id"])
        if experiment is None:
            experiment_id = mlflow.create_experiment(config["experiment_id"])
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_id=experiment_id)
        mlflow.start_run()
        
        logger.info(f"MLFLOW initialized at {config['URI']}")
        logger.info(f"MLFLOW experiment_id set to {experiment_id}")
    else:
        logger.info("MLFLOW not in use")


def mlflow_log_metric(metric: str, value: float, step, config: dict):
    # TODO if mlflow not used log values to logs
    if config["mlflow"]["is_used"]:
        mlflow.log_metric(metric, value, step)
        

def mlflow_log_artifact(tmpfile_name: str, artifact_path: str, config: dict):
    # TODO if mlflow not used iamges to path
    if config["mlflow"]["is_used"]:
        mlflow.log_artifact(tmpfile_name, artifact_path=artifact_path)


def mlflow_log_params(params: dict, config: dict):
    if config["mlflow"]["is_used"]:
        mlflow.log_params(params)


def mlflow_log_summary(gen_net, dis_net, config):
    if config["mlflow"]["is_used"]:
        summary_file = "model_summaries.txt"
        with open(summary_file, "w") as f:
            # Write Generator summary
            f.write("Generator Model Summary:\n")
            f.write("=" * 50 + "\n")
            f.write(str(summary(gen_net, input_size=(1, config["network"]["generator"]['latent_dim']), device='cpu')) + "\n\n")
            f.write("Discriminator Model Summary:\n")
            f.write("=" * 50 + "\n")
            f.write(str(summary(dis_net, input_size=(1,config["data"]["num_channels"],1, config["data"]["seq_len"]), device='cpu')) + "\n\n")

        mlflow_log_artifact(summary_file, None, config)


def set_device():#gen_net: nn.Module, dis_net: nn.Module):
    # TODO add info about choosing GPU
    if not torch.cuda.is_available():
        logger.info('Using CPU, this will be slow')
        device = torch.device("cpu")
    else:
        logger.info('Using default GPU')
        device = torch.device("cuda")
   
    return device


def weights_init(init_type: str, module: nn.Module):

    if isinstance(module, (nn.Conv2d)):
        if init_type == "normal":
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif init_type == "orth":
            nn.init.orthogonal_(module.weight)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight, 1.)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
        else:
            logger.error(f"{init_type} unknown inital type")
            raise NotImplementedError(f"{init_type} unknown inital type")
        
        logger.info(f"Network init with {init_type} type, Layer: {module}")

    elif isinstance(module, ( nn.BatchNorm2d)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)

        logger.info(f"Network init, Layer: {module}")
    
    
def convert_to_tsfresh_format(ts_array):
 
    """
    Convert a 2D array of shape (num_sequences, sequence_length)
    to a DataFrame in tsfresh format: columns = ["id", "time", "value"]
    """
    num_seq, seq_len = ts_array.shape
    df = pd.DataFrame({
        "id": np.repeat(np.arange(num_seq), seq_len),
        "time": np.tile(np.arange(seq_len), num_seq),
        "value": ts_array.flatten()
    })
    return df


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
#             p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            p.data.copy_(cpu_p.cuda().to("cpu"))
            del cpu_p
    
    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)
