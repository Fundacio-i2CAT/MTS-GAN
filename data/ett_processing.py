""" ETT dataset processing pipeline """
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import h5py


class Preprocessing():
    """ Preprocessing class """
    def __init__(self, config):
        self.data_path = config.data_path
        self.saving_path = config.saving_path
        self.train_proportion = config.train_proportion
        self.test_proportion = config.test_proportion
        self.seed = config.seed
        self.seq_len = config.seq_len
        self.overlap = config.overlap

        data = self.load_data()

        self.x_train, self.x_val, self.x_test = self.split_data(data)

        self.x_train, self.y_train = self.create_sequences(self.x_train, seq_len=self.seq_len, overlap=self.overlap)
        self.x_val, self.y_val = self.create_sequences(self.x_val, seq_len=self.seq_len, overlap=self.overlap)
        self.x_test, self.y_test = self.create_sequences(self.x_test, seq_len=self.seq_len, overlap=self.overlap)

        print(f'Train shape: {self.x_train.shape}, Validation shape: {self.x_val.shape}, Test shape: {self.x_test.shape}')

        self.save_data(self.x_train, self.x_val, self.x_test)

    def load_data(self):
        """ Load data from file """
        data = pd.read_csv(os.path.join(self.data_path))
        df = data.sort_values(by='date').reset_index(drop=True)
        return df

    def create_sequences(self, df, seq_len, overlap):
        """ Create sequences with overlap """
        step = seq_len - overlap
        sequences = []
        labels = []
        for i in range(0, len(df) - seq_len + 1, step):
            seq = df.iloc[i:i + seq_len].drop(columns=["date"])
            y = seq["OT"]
            x = seq.drop(columns=["OT"])
            sequences.append(x.to_numpy())
            labels.append(y.to_numpy())

        return np.array(sequences), np.array(labels)

    def split_data(self, data):
        """ Split data into train test and val """
        x_train, x = train_test_split(data, test_size=(1 - self.train_proportion),
                                      random_state=self.seed, shuffle=False)
        x_val, x_test = train_test_split(x, test_size=0.5, random_state=self.seed, shuffle=False)
        return x_train, x_val, x_test

    def save_data(self, train, val, test):
        """ Preprocessing class """
        filename = f'{os.path.splitext(os.path.basename(self.data_path))[0]}_seq_len{self.seq_len}_overlap{self.overlap}.h5'

        with h5py.File(os.path.join(self.saving_path, filename), 'w') as f:
            f.create_dataset('train', data=train)
            f.create_dataset('val', data=val)
            f.create_dataset('test', data=test)
        print(f"Dataset saved in path:{self.saving_path} as {filename}")


if __name__ == "__main__":

    # Labels are not saved in h5 file for ETT
    for file in os.listdir('data/ETT'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str, default=f'data/ETT/{file}')
        parser.add_argument('--saving_path', type=str, default='data/ETT/')
        parser.add_argument('--train_proportion', type=float, default=0.6)
        parser.add_argument('--test_proportion', type=float, default=0.2)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--seq_len', type=int, default=96)
        parser.add_argument('--overlap', type=int, default=86)
        # Biggger overlap, more samples. It cannot be bigger than seq_len

        args = parser.parse_args()
        prep = Preprocessing(args)
