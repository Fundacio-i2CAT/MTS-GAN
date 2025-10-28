"""Predictive and Discriminative evaluation"""
import os
import mlflow
import torch
import numpy as np
import h5py
import predictive as pred
import discriminative as disc
import utils as ut
from utils.load_synthetic import SyntheticGenerator
from data.dataLoader import CustomDataset


def discriminative_predictive_evaluation(n_repetitions=2, tracking_uri='http://0.0.0.0:5004',
                                         run_id="f1bc0bef91394a86af291387a102a53a", epoch=-1):
    """
    Gets config from artifacts in mlflow, reads real data from dataset in config,
    generates synthetic data and computes predictive and discriminative metrics.

    """
    ut.mlflow_init({"URI": "http://0.0.0.0:5004",
                    "experiment_id": "pred_disc_eval",
                    "is_used": True,
                    "run_name": "ib-sh"
                    })
    syn_generator = SyntheticGenerator(run_id=run_id, tracking_uri=tracking_uri, epoch=epoch)

    config = ut.load_yaml(syn_generator.config)
    ut.mlflow_log_param("model", config["type"], None, config)
    ut.mlflow_log_param("dataset", os.path.basename(config['data']['dataset_path']), None, config)
    metrics = {"p_mse": [],
               "d_score": [],
               "d_acc": [],
               }

    for _ in range(n_repetitions):
        mlflow.start_run(nested=True)

        with h5py.File(config['data']['dataset_path'], 'r') as f:
            train_data = f['train'][:]
            validation_data = f['val'][:]
            test_data = f['test'][:]

        train_data = train_data[:, :config["data"]["seq_len"], :]
        validation_data = validation_data[:, :config["data"]["seq_len"], :]
        test_data = test_data[:, :config["data"]["seq_len"], :]

        train_set = CustomDataset(train_data,
                                  norm=config['data']['normalization'])
        validation_set = CustomDataset(validation_data,
                                       norm=config['data']['normalization'], stats=train_set.stats)
        test_set = CustomDataset(test_data,
                                 norm=config['data']['normalization'], stats=train_set.stats)

        _ = np.array([train_set[i] for i in range(len(train_set))])
        validation = np.array([validation_set[i] for i in range(len(validation_set))])
        test = np.array([test_set[i] for i in range(len(test_set))])

        syn_samples = syn_generator.generate_samples(len(validation_set))

        extra_samples = syn_generator.generate_samples(len(test_set))

        if config["type"] == "tts-gan":
            syn_samples = syn_samples.squeeze(2)
            syn_samples = syn_samples.transpose(0, 2, 1)
            extra_samples = extra_samples.squeeze(2)
            extra_samples = extra_samples.transpose(0, 2, 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mse = pred.predictive_score_metrics(test, syn_samples, device)

        ut.mlflow_log_metric("p_mse", mse, None, config)

        acc, discriminative_score = disc.discriminative_score_metrics(validation, test, np.concatenate((syn_samples, extra_samples), axis=0), device)

        ut.mlflow_log_metric("d_score", discriminative_score, None, config)
        ut.mlflow_log_metric("d_acc", acc, None, config)

        metrics['p_mse'].append(mse)
        metrics['d_score'].append(discriminative_score)
        metrics['d_acc'].append(acc)

        print("Discriminative accuracy: ", acc, "\n Discriminative score: ", discriminative_score)
        mlflow.end_run()
    _ = [ut.mlflow_log_metric(key, np.mean(value), None, config) for key, value in metrics.items()]
    _ = [ut.mlflow_log_metric(f'{key}_std', np.std(value), None, config) for key, value in metrics.items()]

    mlflow.end_run()


if __name__ == "__main__":

    b = [

        # ("e76e808c059e44b89791cfbfceadb1fd", 270),
        # ("70052fc8deb545919da5dc1c6fe37563", 200),
        # ("602e1291dbd142b99e9ec53c1cb79c75", 440),
        # ("3e9c8e0c5ab441459663dd407988ce64", 400),
        # ("39dbe7d822f14e1c888fa861b2991692", 480),
        # ("59ebb750e2c04e17a5ee7acbc4fe36c3", 420),
        # ("a141f4383fa549a589ad71a668a35f8d", 420),
        # ("83881951cbe24217a55d21f543afa425", 280),
        # ("02e1d868e3d24a3282869c9636dfa8fa", 210),

    ]

    for i, e in b:
        discriminative_predictive_evaluation(n_repetitions=5, run_id=i, epoch=e)
