""" Unimib SHar dataset processing pipeline """
import os
import argparse
import h5py
import numpy as np
from scipy.io import loadmat


class Preprocessing():
    """ Preprocessing class """
    def __init__(self, config):
        self.data_path = config.data_path
        self.saving_path = config.saving_path
        self.activity_map = config.activity
        self.train_subj = config.train_users
        self.test_subj = config.test_users
        self.val_subj = config.val_users

        data, labels, subjects = self.load_data()
        activity_names = [
            'StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping',
            'GoingDownS', 'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight',
            'FallingBack', 'HittingObstacle', 'FallingWithPS', 'FallingBackSC',
            'Syncope', 'FallingLeft'
        ]

        target_indices = [activity_names.index(self.activity_map) + 1]

        mask = np.isin(labels, target_indices)

        self.x_data = data.reshape((-1, 151, 3))

        self.x_train, self.x_val, self.x_test = self.split_data(subjects, mask)

        self.save_data(self.x_train, self.x_val, self.x_test)

    def load_data(self):
        """ Load data from file """

        data = loadmat(os.path.join(self.data_path, 'adl_data.mat'))['adl_data']
        labels = loadmat(os.path.join(self.data_path, 'adl_labels.mat'))['adl_labels']

        action_num = (labels[:, 0])
        subject_num = (labels[:, 1])  # subject numbers are in column 1 of label

        return data, action_num, subject_num

    def split_data(self, subjects, mask):
        """ Split data into splits based on users"""

        train_index = np.nonzero(np.isin(subjects, self.train_subj))
        test_index = np.nonzero(np.isin(subjects, self.test_subj))
        val_index = np.nonzero(np.isin(subjects, self.val_subj))

        x_train = self.x_data[train_index]
        x_test = self.x_data[test_index]
        x_val = self.x_data[val_index]

        mask_test = mask[test_index]
        mask_train = mask[train_index]
        mask_val = mask[val_index]

        x_test = x_test[mask_test]
        x_train = x_train[mask_train]
        x_val = x_val[mask_val]

        return x_train, x_val, x_test

    def save_data(self, train, val, test):
        """ Save data into new h5 file"""
        filename = f'{self.activity_map}__2.h5'

        with h5py.File(os.path.join(self.saving_path, filename), 'w') as f:
            f.create_dataset('train', data=train)
            f.create_dataset('val', data=val)
            f.create_dataset('test', data=test)
        print(f"Dataset saved in path:{self.saving_path} as {filename}")


if __name__ == "__main__":

    for i in ['Jumping', 'Running',
              # 'StandingUpFS', 'StandingUpFL', 'Walking', 'GoingUpS',
              # 'GoingDownS', 'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight',
              # 'FallingBack', 'HittingObstacle', 'FallingWithPS', 'FallingBackSC',
              # 'Syncope', 'FallingLeft'
              ]:

        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str, default='data/UniMiB-SHAR/UniMiB-SHAR_raw/')
        parser.add_argument('--saving_path', type=str, default='data/UniMiB-SHAR/')
        parser.add_argument('--activity', type=str, default=i)
        parser.add_argument('--train_users', default=[4, 5, 7, 8, 11, 12, 15, 19, 21, 22, 26, 27, 1, 9, 16, 23, 25, 28])
        parser.add_argument('--test_users', default=[2, 3, 13, 17, 18, 30])
        parser.add_argument('--val_users', default=[6, 10, 14, 20, 24, 29])

        args = parser.parse_args()
        prep = Preprocessing(args)
