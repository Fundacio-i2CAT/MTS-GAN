import utils.utils as ut
import yaml
import itertools
import os


# A donde quiero llegar
#param_grid = {   
#    "patch_size": [2, 4, 6, 10],
#    "latent_dim": [50, 100, 150, 200],
#    "embed_dim": [32, 64, 96],
#    "depth": [2, 3, 4, 5],
#    "num_heads": [4, 8, 10],
#    "num_layers": [1,2,3,6,10],
#    "g_lr": [0.00005, 0.0001, 0.0002, 0.0005],
#    "d_lr": [0.00005, 0.0001, 0.0002, 0.0005],
#    "n_critic": [1],
#    "g_scheduler_name": ["linear", "constant", "exponential"],
#    "d_scheduler_name": ["linear", "constant", "exponential"],
#    "loss": ["standard", "hinge", "lsgan", "wgangp"],
#    "epochs": [500, 1000]
#}


config_path = "use_cases/ideko/config.yaml"
base_config = ut.load_yaml(config_path)

dataset = "not_anomalous"
port = "5006"
loss_used = "wgangp" # "standard", "hinge", "lsgan", "wgangp"
periods_taken = 1
total_periods = 3
subsampling = 1
seq_len = int(18000*(periods_taken/total_periods)/subsampling)
experiment_id = f"{loss_used}{subsampling}"

param_grid = { 
    "patch_size": [16, 30, 60, 120],
    "latent_dim": [150],
    "embed_dim": [64, 128, 192],
    "depth": [2, 4, 6],
    "num_heads": [4],

    "g_lr": [0.00005],
    "d_lr": [0.0001],
    "n_critic": [1],
    "g_scheduler_name": ["linear"],
    "d_scheduler_name": ["linear"],
    "g_loss": [loss_used],
    "d_loss": [loss_used],

    "epochs": [500],
}

root_path_save = f"/data/ideko_david/GSconfigs/{experiment_id}"
os.makedirs(root_path_save, exist_ok=True)
grid_search_configs = list(itertools.product(*param_grid.values()))

# TODO: Borrar todo lo que había en esa carpeta para limpiar configs

for i, values in enumerate(grid_search_configs):
    config = base_config.copy()
    
    # Structure parameters
    config["network"]["generator"]["patch_size"] = values[0]
    config["network"]["discriminator"]["patch_size"] = values[0]
    config["network"]["generator"]["latent_dim"] = values[1]
    config["network"]["generator"]["embed_dim"] = values[2]
    config["network"]["generator"]["depth"] = values[3]
    config["network"]["discriminator"]["depth"] = values[3]
    config["network"]["generator"]["num_heads"] = values[4]

    # Optimization parameters
    config["trainer"]["g_lr"] = values[5]
    config["trainer"]["d_lr"] = values[6]
    config["trainer"]["n_critic"] = values[7]
    config["trainer"]["scheduler"]["g_scheduler"]["scheduler_name"] = values[8]
    config["trainer"]["scheduler"]["d_scheduler"]["scheduler_name"] = values[9]
    config["trainer"]["loss"]["g_loss"]["loss_name"] = values[10]
    config["trainer"]["loss"]["d_loss"]["loss_name"] = values[11]

    config["trainer"]["epochs"] = values[12]

    # BASIC params
    config["data"]["seq_len"] = seq_len
    config["network"]["generator"]["seq_len"] = seq_len
    config["network"]["discriminator"]["seq_length"] = seq_len
    config["mlflow"]["experiment_id"] = experiment_id
    config["mlflow"]["URI"] = f"http://0.0.0.0:{port}"
    config["data"]["train_path"] = f"/data/ideko_david/raw_data_splitted/{dataset}/{periods_taken}_periods/{subsampling}_subs/60_train"
    config["data"]["val_path"] = f"/data/ideko_david/raw_data_splitted/{dataset}/{periods_taken}_periods/{subsampling}_subs/20_val"
    config["data"]["test_path"] = f"/data/ideko_david/raw_data_splitted/{dataset}/{periods_taken}_periods/{subsampling}_subs/20_test"

    
    # Save new config file
    config_filename = os.path.join(root_path_save, f"config_{i}.yaml")
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Generated {config_filename}")
