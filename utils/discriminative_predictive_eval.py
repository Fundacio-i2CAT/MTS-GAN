
import numpy as np
from LoadSynthetic import Synthetic_Generator
import h5py
import utils as ut
from data.dataLoader import CustomDataset
import predictive as pred
import discriminative as disc
import torch


def discriminative_predictive_evaluation(tracking_uri='http://0.0.0.0:5003', run_id="f1bc0bef91394a86af291387a102a53a"):
    """
    Gets config from artifacts in mlflow, reads real data from dataset in config,
    generates synthetic data and computes predictive and discriminative metrics.

    """
    syn_generator = Synthetic_Generator(run_id=run_id, tracking_uri=tracking_uri)

    config = ut.load_yaml(syn_generator.config)
    with h5py.File(config['data']['dataset_path'], 'r') as f:
        train_data = f['train'][:]
        validation_data = f['val'][:]
        test_data = f['test'][:]

    train_data = train_data[:, :config["data"]["seq_len"], :]
    validation_data = validation_data[:, :config["data"]["seq_len"], :]
    test_data = test_data[:, :config["data"]["seq_len"], :]

    train_set = CustomDataset(train_data, norm=config['data']['normalization'])
    validation_set = CustomDataset(validation_data, norm=config['data']['normalization'], stats=train_set.stats)
    test_set = CustomDataset(test_data, norm=config['data']['normalization'], stats=train_set.stats)

    validation = np.array([validation_set[i] for i in range(len(validation_set))])
    test = np.array([test_set[i] for i in range(len(test_set))])

    syn_samples = syn_generator.generate_samples(len(validation_set))
    syn_samples = syn_samples.squeeze(2)
    syn_samples = syn_samples.transpose(0, 2, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mse = pred.predictive_score_metrics(test, syn_samples, device)
    baseline_mse = pred.predictive_score_metrics(test, validation, device)

    discriminative_acc = disc.discriminative_score_metrics(test, syn_samples, device)

    print("Trained on real data: ", baseline_mse, "\n Trained on synthetic data:", mse)

    print("Discriminative accuracy: ", discriminative_acc)


if __name__ == "__main__":
    discriminative_predictive_evaluation()
