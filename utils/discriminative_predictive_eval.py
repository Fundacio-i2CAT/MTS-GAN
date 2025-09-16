
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


def discriminative_predictive_evaluation(n_repetitions=2, tracking_uri='http://0.0.0.0:5004', run_id="f1bc0bef91394a86af291387a102a53a", epoch=-1):
    """
    Gets config from artifacts in mlflow, reads real data from dataset in config,
    generates synthetic data and computes predictive and discriminative metrics.

    """
    ut.mlflow_init({"URI": "http://0.0.0.0:5004",
                    "experiment_id": "pred_disc_eval_tts",
                    "is_used": True,
                    "run_name": "ib-sh"
                    })
    syn_generator = Synthetic_Generator(run_id=run_id, tracking_uri=tracking_uri, epoch=epoch)

    config = ut.load_yaml(syn_generator.config)
    ut.mlflow_log_param("model", "tts", None, config)
    ut.mlflow_log_param("dataset", os.path.basename(config['data']['dataset_path']), None, config)
    metrics = {"p_mse": [],
               "d_score": [],
               "d_acc": [],
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
        #baseline_mse = pred.predictive_score_metrics(test, validation, device)

        ut.mlflow_log_metric("p_mse", mse, None, config)
        #ut.mlflow_log_metric("p_mse_baseline", baseline_mse, None, config)

        acc, discriminative_score = disc.discriminative_score_metrics(validation, test, np.concatenate((syn_samples, extra_samples), axis=0), device)
        #acc_baseline, discriminative_score_baseline = disc.discriminative_score_metrics(validation, test, train, device)

        ut.mlflow_log_metric("d_score", discriminative_score, None, config)
        ut.mlflow_log_metric("d_acc", acc, None, config)

        #ut.mlflow_log_metric("d_score_baseline", discriminative_score_baseline, None, config)
        #ut.mlflow_log_metric("d_acc_baseline", acc_baseline, None, config)

        metrics['p_mse'].append(mse)
        #metrics['p_mse_baseline'].append(baseline_mse)
        metrics['d_score'].append(discriminative_score)
        metrics['d_acc'].append(acc)
        #metrics['d_score_baseline'].append(discriminative_score_baseline)
        #metrics['d_acc_baseline'].append(acc_baseline)

        #print("Trained on real data: ", baseline_mse, "\n Trained on synthetic data:", mse)

        print("Discriminative accuracy: ", acc, "\n Discriminative score: ", discriminative_score)
        mlflow.end_run()
    _ = [ut.mlflow_log_metric(key, np.mean(value), None, config) for key, value in metrics.items()]
    _ = [ut.mlflow_log_metric(f'{key}_std', np.std(value), None, config) for key, value in metrics.items()]

    mlflow.end_run()


if __name__ == "__main__":

    b = [
    #("def93d77a62d43f297c543473920b6cd", 440),
    ("eee11e8d63624eea8a9797e9661d9810", 320)
    ]
    
    for i,e in b:
        discriminative_predictive_evaluation(n_repetitions=5, run_id=i, epoch=e)
