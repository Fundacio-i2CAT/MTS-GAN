# Class for inference and create samples
# Made them to a Pytorch Dataset 

from torch.utils.data import Dataset
import torch
from architecture.generator import Generator
import numpy as np
import mlflow
import re


class Synthetic_Generator_ideko(Dataset):
    def __init__(self, 
                 run_id='c576f70a3f4541949ed1ee4dac244893',
                 tracking_uri='http://0.0.0.0:5010'
                 ):

        self.run_id = run_id
        self.tracking_uri = tracking_uri
        
        # Load generator checkpoint from MLflow
        latest_checkpoint_path = self._get_latest_checkpoint_path(run_id)
        self.gen_ckp = self._load_checkpoint(run_id, latest_checkpoint_path)
        self.gen_net = Generator(self.gen_ckp['model_config']['generator'])
        self.gen_net.load_state_dict(self.gen_ckp['gen_state_dict'])
 
    def generate_samples(self, samples: int):
        z = torch.FloatTensor(np.random.normal(0, 1, (samples, 100)))
        self.syn_sample = self.gen_net(z)
        self.syn_sample = self.syn_sample.detach().numpy()

        return self.syn_sample
 
    def _get_latest_checkpoint_path(self, run_id: str):
        """Find the latest checkpoint file (highest-numbered) in the 'checkpoints/' folder."""
        mlflow.set_tracking_uri(self.tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # List artifacts in the "checkpoints" folder
        artifacts = client.list_artifacts(run_id, path="checkpoints")
        checkpoint_files = [artifact.path for artifact in artifacts if artifact.path.endswith(".checkpoint")]

        # Extract numbers from filenames like "235_checkpoint.pth"
        checkpoint_numbers = []
        pattern = re.compile(r"(\d+)_checkpoint\.checkpoint")

        for file in checkpoint_files:
            match = pattern.match(file.split("/")[-1])
            if match:
                checkpoint_numbers.append(int(match.group(1)))

        if not checkpoint_numbers:
            raise ValueError("No valid checkpoint files found in MLflow artifacts.")

        # Find the file with the highest number (latest epoch)
        latest_checkpoint = max(checkpoint_numbers)
        latest_checkpoint_file = f"checkpoints/{latest_checkpoint}_checkpoint.checkpoint"
        return latest_checkpoint_file

    def _load_checkpoint(self, run_id: str, checkpoint_path: str):
        """Load the generator model checkpoint from a given MLflow artifact path."""
        mlflow.set_tracking_uri(self.tracking_uri)
        local_checkpoint_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{checkpoint_path}")
        checkpoint = torch.load(local_checkpoint_path, map_location=torch.device('cpu'))
        return checkpoint
        
    
