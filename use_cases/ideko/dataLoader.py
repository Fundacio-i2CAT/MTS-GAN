
import os
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
 

class IdekoDatasetLoader(Dataset):
    def __init__(self, config: dict, data_mode: str = "Train"):

        logger_deep = logging.getLogger('DataLogger')
        self.is_normalize = config.get("normalization", "min_max")
        self.data_path = config.get("dataset_path")
        self.seq_len = config.get("seq_len", 300)  
        self.num_channels = config.get("num_channels", 2) 
        self.y_label = 1       # setting a default label value, could be customized
        self.split_proportion = config.get("split_proportion", 0.2)
        self.seed = config.get("seed", None)
        self.data_mode = data_mode
         
        # Load and prepare data
        self.x_data = self._load_data()
        self.x_train, self.x_test = self._split_data()
        
        self.x_train = self._reshape_data(self.x_train)
        self.x_test = self._reshape_data(self.x_test)

        # Normalize if required
        if self.is_normalize == "standarize":
            self.x_train = self.normalization(self.x_train)
            self.x_test = self.normalization(self.x_test)

        if self.is_normalize == "min_max":
            self.x_train = self.min_max_normalization(self.x_train)
            self.x_test = self.min_max_normalization(self.x_test)

        logger_deep.info(f"x_train shape after reshape: {self.x_train.shape}")
        logger_deep.info(f"x_validation shape after reshape: {self.x_test.shape}")
 
    def _load_data(self):
 
        data_samples = []

        for file in os.listdir(self.data_path):
            if file.endswith('.csv'):
                file_path = os.path.join(self.data_path, file)
                data = pd.read_csv(file_path)
                if 'time' in data.columns:
                    data = data.drop(columns=['time']) 
                    data = data.drop(columns=['f1']) 
                    data = data.drop(columns=['f4']) 
                data = data.values 
                if data.shape[0] < self.seq_len:
                    
                    padding = np.zeros((self.seq_len - data.shape[0], data.shape[1]))
                    data = np.vstack((data, padding))
                elif data.shape[0] > self.seq_len:
                    # Truncate if rows exceed seq_len
                    data = data[:self.seq_len, :]

                data_samples.append(data)
 
        x_data = np.stack(data_samples)
        return x_data
    
    def _split_data(self):
 
        # Split data into train (60%) and remaining (40%)
        x_train, x_test = train_test_split(self.x_data, test_size=self.split_proportion, random_state=self.seed)
        
        return x_train, x_test 
 
    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]       

    def _min_max_normalize(self, epoch):
        
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result

    def _normalize(self, epoch):
        """Normalize each channel independently in a sample."""
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / (np.sqrt(epoch.var(axis=0)) + e)
        return result
        
    def _reshape_data(self, data):
        """Reshape data from (num_samples, seq_len, num_channels) to (num_samples, num_channels, 1, seq_len)."""
        data = np.transpose(data, (0, 2, 1))  # Reorder to (num_samples, num_channels, max_rows)
        data = data.reshape(data.shape[0], self.num_channels, 1, self.seq_len)  # Add singleton dimension
        return data
    
    def normalization(self, epochs):
        """Apply normalization to each sample and channel."""
        for i in range(epochs.shape[0]):
 
            for j in range(epochs.shape[1]):
 
                if j == 0:
                    epochs[i, j, 0, :] = self._normalize(epochs[i, j, 0, :]) 
                else:
                    epochs[i, j, 0, :] = self._normalize(epochs[i, j, 0, :]) 
        return epochs
    
    def min_max_normalization(self, epochs):
        """Apply min_max normalization to each sample and channel."""
        for i in range(epochs.shape[0]):
 
            for j in range(epochs.shape[1]):
 
                if j == 0:
                    epochs[i, j, 0, :] = self._min_max_normalize(epochs[i, j, 0, :]) 
                else:
                    epochs[i, j, 0, :] = self._min_max_normalize(epochs[i, j, 0, :]) 
        return epochs
    
    def __getitem__(self, idx):
        """Returns a sample based on the data mode (Train, Validation)."""
        if self.data_mode == 'Train':
            return self.x_train[idx], self.y_label
        elif self.data_mode == 'Validation':
            return self.x_test[idx], self.y_label
        else:
            raise ValueError("data_mode must be 'Train', or 'Validation'")
 
    def __len__(self):
        """Returns the length of the dataset based on the data mode."""
        if self.data_mode == 'Train':
            return len(self.x_train)
        elif self.data_mode == 'Validation':
            return len(self.x_test)
            
    