import os
import argparse
import pandas as pd
from torch.utils import data
from utils.load_synthetic import SyntheticGenerator


def generate_synthetic_data(num_samples, tracking_uri='http://0.0.0.0:5004',
                            run_id="f1bc0bef91394a86af291387a102a53a"):
    """Main Synthetic Data generation function"""
    syn_generator = SyntheticGenerator(run_id=run_id, tracking_uri=tracking_uri)
    syn_samples = syn_generator.generate_samples(num_samples)
    return syn_samples


def save_synthetic_data(data_syn, run_id, base_dir="plots/data/syn_data"):
    """Save synthetic data samples to CSV files with feature names f1, f2, ..."""
    save_dir = os.path.join(base_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    # Expecting data shape: [num_sequences, sequence_len, features]
    num_features = data_syn.shape[2]  # features dimension
    feature_names = [f"f{i+1}" for i in range(num_features)]

    for i, sample in enumerate(data_syn):
        df = pd.DataFrame(sample, columns=feature_names)
        save_path = os.path.join(save_dir, f"synthetic_sample_{i}.csv")
        df.to_csv(save_path, index=False)
    
    
    print(f"Saved {len(data_syn)} samples to: {save_dir}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Synthetic Data Inference Script")

    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="http://0.0.0.0:5004",
        help="Tracking URI for MLflow or equivalent service"
    )

    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID to load the trained model from"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of synthetic samples to generate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    plot_path = "data/syn_data"
    os.makedirs(plot_path, exist_ok=True)

    # Generate data
    data_syn = generate_synthetic_data(
        num_samples=args.num_samples,
        tracking_uri=args.tracking_uri,
        run_id=args.run_id
    )

    # Ensure shape [sequence, sequence_len, features]
    if len(data_syn.shape) >= 4:
        data_syn = data_syn.squeeze(2)
        data_syn = data_syn.transpose(0, 2, 1)

    # Create DataLoader (optional, if needed downstream)
    syn_data_loader = data.DataLoader(
        data_syn,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    # Save data
    save_synthetic_data(data_syn, args.run_id)