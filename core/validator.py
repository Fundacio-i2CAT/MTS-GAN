import torch
from tqdm import tqdm
import numpy as np
import logging
from torch.utils import data
import matplotlib.pyplot as plt
import os

import utils.metrics as metrics
import utils.utils as ut

logger = logging.getLogger("DataLogger")


def validate(gen_net: torch.nn.Module, validation_loader: data.DataLoader, config: dict, tracker: dict, device) -> dict:
 
    gen_net.eval()
    real_samples = []
    fake_samples = []
    global_steps = tracker['validation_global_step']
 
    with torch.no_grad():

        for iter_idx, ts in enumerate(tqdm(validation_loader, desc="Validation Progress", unit="batch")):
            real_ts = ts.to(device, dtype=torch.float, non_blocking=True)
            z = torch.randn(ts.shape[0], config["network"]["generator"]["latent_dim"], device=device)
            fake_ts = gen_net(z)

            real_ts = real_ts.cpu().numpy()
            fake_ts = fake_ts.cpu().numpy()

            # real_ts (num_seq, feat, 1, len_seq)
            real_ts = real_ts.squeeze(2).transpose(0, 2, 1)  # Remove the singleton dimension and trasnpose -> (num_seq, len_seq,  feat)
            fake_ts = fake_ts.squeeze(2).transpose(0, 2, 1)

            real_samples.append(real_ts)
            fake_samples.append(fake_ts)
            
    real_data = np.concatenate(real_samples, axis=0)
    generated_data = np.concatenate(fake_samples, axis=0)
    validation_metrics = metrics.compute_metrics(real_data, generated_data)
    [ut.mlflow_log_metric(key, value, step=tracker['validation_global_step'], config=config) for key, value in validation_metrics.items()]

    tracker['validation_global_step'] = global_steps + 1
 
    return validation_metrics


def gen_plot(gen_net, epoch, config):

    synthetic_data = []

    device = next(gen_net.parameters()).device

    for i in range(6):
        fake_noise = torch.randn(1, config["network"]["generator"]["latent_dim"], device=device)
        fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
        synthetic_data.append(fake_sigs)
        
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Synthetic samples at epoch {epoch}', fontsize=20)

    for row, feature_index in enumerate(range(2)): 
        for col in range(3): 
         
            data = synthetic_data[row * 3 + col]
            feature_data = data[0, feature_index, 0, :] 
            axs[row, col].plot(feature_data)
            axs[row, col].set_title(f'Feature f{feature_index + 1}, Plot {col + 1}', fontsize=15)
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel(f'f{feature_index + 1}')

    plt.subplots_adjust(hspace=0.4)
    temp_filename = f"{epoch}_epoch.jpeg"
    temp_dir = "/data/tmp_extreme_xp/"  
    tmpfile_name = os.path.join(temp_dir, temp_filename)
    plt.savefig(tmpfile_name, format='jpeg')
    plt.clf()
    ut.mlflow_log_artifact(tmpfile_name, artifact_path="synthetic_signals", config=config)
    plt.close()
    os.remove(tmpfile_name)