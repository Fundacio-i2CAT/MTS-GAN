""" Data Loader """
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
        Data Loader that generates output of shape
        (num_samples, sequence_length, num_features)
        and normalized data
    """

    def __init__(self, x, norm='none', stats=None):

        self.stats = stats
        self.x = self.normalization(x, norm)

    def normalization(self, x, norm):
        """ Normalization function, None, min max, global min max, standard """

        if norm == 'none':
            return x

        if norm == 'min_max':
            min_vals = x.min(axis=1, keepdims=True)
            max_vals = x.max(axis=1, keepdims=True)

            x = (x - min_vals) / (max_vals - min_vals + 1e-8)
            return x

        if norm == 'global_min_max':
            if self.stats is not None:
                min_val, max_val = self.stats['min'], self.stats['max']
            else:
                self.stats = {}
                min_val = x.min(axis=(0, 1))  # shape (2,)
                max_val = x.max(axis=(0, 1))

                self.stats['min'], self.stats['max'] = min_val, max_val

            x = (x - min_val) / (max_val - min_val + 1e-8)
            return x

        if norm == 'standarization':
            if self.stats is not None:
                mean, std = self.stats['mean'], self.stats['std']
            else:
                self.stats = {}
                mean = x.mean(axis=1, keepdims=True)
                std = x.std(axis=1, keepdims=True)
                self.stats['mean'], self.stats['std'] = mean, std

            return (x - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
