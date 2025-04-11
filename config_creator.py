import utils.utils as ut
import yaml
import itertools
import os

# Lo que había:
#param_grid = {
#    "patch_size": [2],
#    "latent_dim": [50, 100, 150, 200],
#    "embed_dim": [10],
#    "depth": [3],
#    "num_heads": [10],
#    "g_lr": [0.0001],
#    "d_lr": [0.00005],
#    "n_critic": [1],
#    "g_scheduler_name": ["linear"],
#    "d_scheduler_name": ["linear"],
#    "loss": ["lsgan"],
#    "num_layers": [1,2,3,6,10]
#}

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
root_path_save = "/data/ideko_david/GSconfigs/optimization_params"
os.makedirs(root_path_save, exist_ok=True)

param_grid = {
    "patch_size": [2],
    "latent_dim": [150],
    "embed_dim": [64],
    "depth": [2],
    "num_heads": [4],

    "g_lr": [0.0001],
    "d_lr": [0.00005],
    "n_critic": [1],
    "g_scheduler_name": ["linear", "constant", "exponential"],
    "d_scheduler_name": ["linear", "constant", "exponential"],
    "loss": ["standard", "hinge", "lsgan", "wgangp"],

}
grid_search_configs = list(itertools.product(*param_grid.values()))


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
    config["trainer"]["loss"]["d_loss"]["loss_name"] = values[10]
    
    # Save new config file
    config_filename = os.path.join(root_path_save, f"config_{i}.yaml")
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Generated {config_filename}")
