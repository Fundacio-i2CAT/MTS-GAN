"""Inference Script"""
import os
from torch.utils import data
from utils.load_synthetic import SyntheticGenerator


def generate_synthetic_data(num_samples, tracking_uri='http://0.0.0.0:5004',
                            run_id="f1bc0bef91394a86af291387a102a53a"):
    """Main Synthethic Data generation function"""
    syn_generator = SyntheticGenerator(run_id=run_id, tracking_uri=tracking_uri)
    syn_samples = syn_generator.generate_samples(num_samples)

    return syn_samples


if __name__ == "__main__":

    plot_path = "plots"
    os.makedirs(plot_path, exist_ok=True)
    runid = "<ENTER RUN ID HERE>"

    data_syn = generate_synthetic_data(10, run_id=runid)
    if len(data_syn.shape) >= 4:
        data_syn = data_syn.squeeze(2)
        data_syn = data_syn.transpose(0, 2, 1)

    syn_data_loader = data.DataLoader(data_syn, batch_size=1, num_workers=1, shuffle=False)
