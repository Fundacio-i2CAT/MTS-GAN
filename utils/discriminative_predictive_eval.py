
import numpy as np
from LoadSynthetic import Synthetic_Generator
import h5py
import utils as ut
from data.dataLoader import CustomDataset
import predictive as pred
import discriminative as disc
import torch
import mlflow
import os


def discriminative_predictive_evaluation(n_repetitions=2, tracking_uri='http://0.0.0.0:5003', run_id="f1bc0bef91394a86af291387a102a53a"):
    """
    Gets config from artifacts in mlflow, reads real data from dataset in config,
    generates synthetic data and computes predictive and discriminative metrics.

    """
    ut.mlflow_init({"URI": "http://0.0.0.0:5003",
                    "experiment_id": "pred_disc_eval",
                    "is_used": True,
                    "run_name": "ib-sh"
                    })
    syn_generator = Synthetic_Generator(run_id=run_id, tracking_uri=tracking_uri)

    config = ut.load_yaml(syn_generator.config)
    ut.mlflow_log_param("model", "tts", None, config)
    ut.mlflow_log_param("dataset", os.path.basename(config['data']['dataset_path']), None, config)
    metrics = {"p_mse": [],
               "p_mse_baseline": [],
               "d_score": [],
               "d_acc": [],
               "d_score_baseline": [],
               "d_acc_baseline": []
               }

    for i in range(n_repetitions):
        mlflow.start_run(nested=True)

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
        
        train = np.array([train_set[i] for i in range(len(train_set))])
        validation = np.array([validation_set[i] for i in range(len(validation_set))])
        test = np.array([test_set[i] for i in range(len(test_set))])

        syn_samples = syn_generator.generate_samples(len(validation_set))
        syn_samples = syn_samples.squeeze(2)
        syn_samples = syn_samples.transpose(0, 2, 1)

        extra_samples = syn_generator.generate_samples(len(test_set))
        extra_samples = extra_samples.squeeze(2)
        extra_samples = extra_samples.transpose(0, 2, 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mse = pred.predictive_score_metrics(test, syn_samples, device)
        baseline_mse = pred.predictive_score_metrics(test, validation, device)

        ut.mlflow_log_metric("p_mse", mse, None, config)
        ut.mlflow_log_metric("p_mse_baseline", baseline_mse, None, config)

        acc, discriminative_score = disc.discriminative_score_metrics(validation, test, np.concatenate((syn_samples, extra_samples), axis=0), device)
        acc_baseline, discriminative_score_baseline = disc.discriminative_score_metrics(validation, test, train, device)

        ut.mlflow_log_metric("d_score", discriminative_score, None, config)
        ut.mlflow_log_metric("d_acc", acc, None, config)

        ut.mlflow_log_metric("d_score_baseline", discriminative_score_baseline, None, config)
        ut.mlflow_log_metric("d_acc_baseline", acc_baseline, None, config)

        metrics['p_mse'].append(mse)
        metrics['p_mse_baseline'].append(baseline_mse)
        metrics['d_score'].append(discriminative_score)
        metrics['d_acc'].append(acc)
        metrics['d_score_baseline'].append(0)
        metrics['d_acc_baseline'].append(0)

        print("Trained on real data: ", baseline_mse, "\n Trained on synthetic data:", mse)

        print("Discriminative accuracy: ", acc, "\n Discriminative score: ", discriminative_score)
        mlflow.end_run()
    _ = [ut.mlflow_log_metric(key, np.mean(value), None, config) for key, value in metrics.items()]
    mlflow.end_run()


if __name__ == "__main__":
    a = ["06952188de164d3db1f879ba7ebe8e0d",  # Jumping
         "d9d1181b9ec54bb7bf6ac3eb0eb2d8f5",  # running
         "f54f5e3931624f879f98c328f0c01de0",  # ETTh2_720
         "0714d344d09e489e80a4b5683dea9452",  # ETTh2_336
         "0b2442e001e34a809cf8910b3beaec6a",  # ETTh2_96
         "a3e773d077cd441491912781e4756fb5",  # ETTh1_720
         "a7571ce140234d88a7a629651d321748",  # ETTh1_336
         "ea315c63cea944adb359dbfc3cb129ad",  # ETTh1_96
         "f1bc0bef91394a86af291387a102a53a",  # not_anomalous
         "2b11c47ae3944acca1d9c77bc0fb6161",  # electrical
         "a12b83bdfab242e7b82e6ef5c6dc8529",  # mechanical
         ]
    for i in a:
        discriminative_predictive_evaluation(n_repetitions=5, run_id=i)
