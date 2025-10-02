import argparse
import itertools
import yaml
import utils.utils as ut
from main import main
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_mamba.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    # main(args)

    config = ut.load_yaml(args.config)
    datasets_ids = [
        #("/data/extremexp/extremexp_data/ideko/not_anomalous_3periods_3subsampling_6000seqLen.h5", 6000, 2, "ttsmamba_ideko_na_6000"),
        #("/data/extremexp/extremexp_data/ideko/electrical_anomalies_filtered_3periods_3subsampling_6000seqLen.h5", 6000, 2, "ttsmamba_ideko_elec_6000"),
        #("/data/extremexp/extremexp_data/ideko/electrical_anomalies_filtered_3periods_60subsampling_300seqLen.h5", 300, 2, "ttsmamba_ideko_elec_300"),
        #("/data/extremexp/extremexp_data/ideko/not_anomalous_3periods_60subsampling_300seqLen.h5", 300, 2, "ttsmamba_ideko_na_300"),
        #("/data/extremexp/extremexp_data/unimib-shar/Jumping.h5", 151, 3, "ttsmamba_jumping"),
        #("/data/extremexp/extremexp_data/unimib-shar/Running.h5", 151, 3, "ttsmamba_running"),
        #("/data/extremexp/extremexp_data/ETT/ETTh1_seq_len96_overlap86.h5", 96, 6, "ttsmamba_ett96"),
        ("/data/extremexp/extremexp_data/ETT/ETTh1_seq_len336_overlap326.h5", 336, 6, "ttsmamba_ett336"),
        #("/data/extremexp/extremexp_data/ETT/ETTh1_seq_len720_overlap710.h5", 720, 6, "ttsmamba_ett720")
    ]


    for data in datasets_ids:
        gridsearch = {
            "normalization": ["global_min_max"],  # , "min_max"],
            "scheduler_name": ["linear"],
            "loss_name": ["lsgan"],
            "epochs": [500],
            "batch_size": [64],

            "lr_g": [0.0003],
            "lr_d": [0.0001],

            "seq_len": [data[1]],
            "features": [data[2]],
            "dataset_path": [data[0]],

            "hidden_dim": [256, 64, 512], # 64, 256, 512
            "d_state": [256, 128, 64], # 256, 512, 128, 64
            "d_conv": [4], # 2, 4
            "e_fact": [2], # 1, 2
            "channels": [3],
            "z_dim": [100],
            "num_layers": [1,3],
            
            "disc_hidden_dim": [256],
            "disc_d_state": [256],
            "disc_d_conv": [2],
            "disc_e_fact": [1],
            "disc_channels": [3],
            "disc_z_dim": [100],
            "disc_num_layers": [1],
        }

        best_jensen = 1
        best_jensen_arch = 1

        hierarchy_combinations = list(itertools.product(*gridsearch.values()))
        formatted_combinations = [
            dict(zip(gridsearch.keys(), combination)) for combination in hierarchy_combinations
        ]

        for combination in formatted_combinations:

            # Default config values
            config["data"]["dataset_path"] = combination['dataset_path']
            config['mlflow']['run_name'] = combination['dataset_path'][23:28]
            config['mlflow']['experiment_id'] = data[3]
            config["data"]["seq_len"] = combination["seq_len"]
            config["data"]["num_channels"] = combination["features"]
            config["network"]["generator"]["seq_len"] = combination["seq_len"]
            config["network"]["generator"]["channels"] = combination["features"]
            config["network"]["discriminator"]["seq_len"] = combination["seq_len"]
            config["network"]["discriminator"]["channels"] = combination["features"]
            # Optimitzation hyperparameters
            config["trainer"]["batch_size"] = combination["batch_size"]
            config["trainer"]["epochs"] = combination["epochs"]

            config["data"]["normalization"] = combination["normalization"]

            config["trainer"]["loss"]["g_loss"]["loss_name"] = combination["loss_name"]
            config["trainer"]["loss"]["d_loss"]["loss_name"] = combination["loss_name"]

            config["trainer"]["scheduler"]["g_scheduler"]["scheduler_name"] = combination["scheduler_name"]
            config["trainer"]["scheduler"]["d_scheduler"]["scheduler_name"] = combination["scheduler_name"]

            config["trainer"]["g_lr"] = combination["lr_g"]
            config["trainer"]["d_lr"] = combination["lr_d"]
            
            config["network"]["generator"]["hidden_dim"] = combination['hidden_dim']
            config["network"]["generator"]["d_state"] = combination['d_state']
            config["network"]["generator"]["d_conv"] = combination['d_conv']
            config["network"]["generator"]["e_fact"] = combination['e_fact']
            config["network"]["generator"]["z_dim"] = combination['z_dim']
            config["network"]["generator"]["num_layers"] = combination['num_layers']

            config["network"]["discriminator"]["hidden_dim"] = combination['hidden_dim']
            config["network"]["discriminator"]["d_state"] = combination['d_state']
            config["network"]["discriminator"]["d_conv"] = combination['disc_d_conv']
            config["network"]["discriminator"]["e_fact"] = combination['disc_e_fact']
            config["network"]["discriminator"]["z_dim"] = combination['disc_z_dim']
            config["network"]["discriminator"]["num_layers"] = combination['disc_num_layers']

            with open("gridsearch.yaml", "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            parser = argparse.ArgumentParser()
            parser.add_argument("--config", type=str, default="gridsearch.yaml",
                                help="Path to YAML config file")
            args = parser.parse_args()

            jensen = main(args)
