
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
from use_case.ideko.LoadSynthetic_ideko import Synthetic_Generator_ideko
from torch.utils import data

plot_save_path = "plots"
os.makedirs(plot_save_path, exist_ok=True)


def save_time_series_plot(array, plot_name):
    print(array.shape)
    reshaped_array = array.squeeze(0).squeeze(1) 
    df = pd.DataFrame({
        'time': np.arange(reshaped_array.shape[1]),
        'f2': reshaped_array[0, :].numpy(),   
        'f3': reshaped_array[1, :].numpy()  
    })
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('f2', color='blue')
    ax1.plot(df['time'], df['f2'], label='f2', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('f3', color='orange')
    ax2.plot(df['time'], df['f3'], label='f3', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.title(f'Synthetic Time Series')
    plot_file_path = os.path.join(plot_save_path, f"{plot_name}.png")
    plt.savefig(plot_file_path)
    plt.clf()
    plt.close()
    gc.collect()

def main():
    # using mlflow
    samples = 10
    run_id = f"cda256441fac4896b2d119a27e799904"
    tracking_uri = 'http://0.0.0.0:5010'
    syn_generator = Synthetic_Generator_ideko(run_id=run_id, tracking_uri=tracking_uri)
    syn_samples = syn_generator.generate_samples(samples)
    syn_data_loader = data.DataLoader(syn_samples, batch_size=1, num_workers=1, shuffle=False)
    
    # Plot and save each category
    for i, sig in enumerate(syn_data_loader):
        save_time_series_plot(sig, f"syn_sample_{i+1}")



if __name__ == "__main__":
    main()