
import os
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset

class IdekoDatasetLoader(Dataset):

    def __init__(self, data_mode, config, min_values=None, max_values=None):
        logger_deep = logging.getLogger('DataLogger')
        self.is_normalize = config["normalization"]
        self.features = config["features"]
        self.num_channels = config["num_channels"]
        self.seq_len = config["seq_len"]
        self.y_label = 1   

        # Loading data
        if data_mode == "train":
            self.data_folder = config["train_path"]
        elif data_mode == "val":
            self.data_folder = config["val_path"]
        else:
            self.data_folder = config["test_path"]
        self.X_data = self._load_data(self.data_folder) 

        # Calc. global maxs/mins
        self.min_values = {f"{x}": int(1e9) for x in self.features}
        self.max_values = {f"{x}": int(0) for x in self.features}

        if data_mode == "train":
            self._calculate_global_mins_maxs()
        else:
            self.min_values = min_values
            self.max_values = max_values

        # Load and prepare data
        self.X_data = self._reshape_data(self.X_data)

        # Normalize if required
        if self.is_normalize == "min_max":
            self.X_data = self.min_max_normalization(self.X_data)

        if self.is_normalize == "global_min_max":
            self.X_data = self.global_min_max_normalization(self.X_data)

        logger_deep.info(f"X_{data_mode} shape after reshape: {self.X_data.shape}")


    def _calculate_global_mins_maxs(self):
        for image in self.X_data:
            for features_info in image:
                for j in range(len(self.features)):
                    feature_name = self.features[j]

                    if self.min_values[feature_name] > features_info[j]:
                        self.min_values[feature_name] = features_info[j]
                    
                    if self.max_values[feature_name] < features_info[j]:
                        self.max_values[feature_name] = features_info[j]


    def _load_data(self, data_path):
        data_samples = []

        for file in os.listdir(data_path):
            if file.endswith('.csv'):
                file_path = os.path.join(data_path, file)
                data = pd.read_csv(file_path)

                not_selected = [col for col in data.columns if col not in self.features]
                if 'time' in data.columns:
                    data = data.drop(columns=not_selected)

                data = data.values
                if data.shape[0] < self.seq_len:
                    padding = np.zeros((self.seq_len - data.shape[0], data.shape[1]))
                    data = np.vstack((data, padding))
                
                # Truncate if rows exceed seq_len
                elif data.shape[0] > self.seq_len:
                    data = data[:self.seq_len, :]

                data_samples.append(data)

        X_data = np.stack(data_samples)
        return X_data      


    def _min_max_normalize(self, epoch):
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result

    def _global_min_max_normalize(self, epoch, feature_name):
        min_val = self.min_values[feature_name]
        max_val = self.max_values[feature_name]

        result = (epoch - min_val) / (max_val - min_val)
        return result
        
    def _reshape_data(self, data):
        """Reshape data from (num_samples, seq_len, num_channels) to (num_samples, num_channels, 1, seq_len)."""
        data = np.transpose(data, (0, 2, 1))  # Reorder to (num_samples, num_channels, max_rows)
        print(data)
        data = data.reshape(data.shape[0], self.num_channels, 1, self.seq_len)  # Add singleton dimension
        print("no peta en el reshape")
        return data
    
    def min_max_normalization(self, epochs):
        """Apply min_max normalization to each sample and channel."""
        for i in range(epochs.shape[0]):
 
            for j in range(epochs.shape[1]):
 
                if j == 0:
                    epochs[i, j, 0, :] = self._min_max_normalize(epochs[i, j, 0, :]) 
                else:
                    epochs[i, j, 0, :] = self._min_max_normalize(epochs[i, j, 0, :]) 
        return epochs
    
    def global_min_max_normalization(self, epochs):
        """Apply normalization to each sample and channel."""
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
 
                if j == 0:
                    feature_name = "f2"
                elif j == 1:
                    feature_name = "f3"
                else:
                    raise ValueError("Hay al menos una feature para normalizar con global_min_max que no se ha contemplado.")
                
                epochs[i, j, 0, :] = self._global_min_max_normalize(epochs[i, j, 0, :], feature_name) 

        return epochs

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_label
