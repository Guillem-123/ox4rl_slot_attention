import torch
from collections import Counter
from ox4rl.dataset.atari_data_collector import AtariDataCollector


class ZWhatDataPreparation:

    def __init__(self, cfg):
        self.cfg = cfg
    
    def collect_and_prepare_z_what_data(self, cfg, dataset_mode, data_subset_modes):        
        data = {}
        for data_subset_mode in data_subset_modes:
            z_whats, labels = AtariDataCollector.collect_z_what_data_reshaped(cfg, dataset_mode, data_subset_mode)
            data[data_subset_mode] = ZWhatDataPreparation.prepare_data(z_whats, labels)
        return data
    
    @staticmethod
    def prepare_data(z_what, labels,):
        c = Counter(labels.tolist() if labels is not None else [])
        relevant_labels = list(c.keys())

        # Filter out the irrelevant labels
        z_what, labels = ZWhatDataPreparation.only_keep_relevant_data(z_what, labels, relevant_labels)
        # Split the data into train and test
        train_x, train_y, test_x, test_y = ZWhatDataPreparation.train_test_split(z_what, labels, train_portion=0.9)

        if len(c) < 2 or len(torch.unique(train_y)) < 2:
            return None, None, None, None, None

        return relevant_labels, test_x, test_y, train_x, train_y
    
    @staticmethod
    def train_test_split(z_what, labels, train_portion=0.9):
        nb_sample = int(train_portion * len(labels))
        train_x = z_what[:nb_sample]
        train_y = labels[:nb_sample]
        test_x = z_what[nb_sample:]
        test_y = labels[nb_sample:]
        return train_x, train_y, test_x, test_y

    @staticmethod
    def only_keep_relevant_data(z_what, labels, relevant_labels):
        relevant_mask = torch.zeros(labels.shape, dtype=torch.bool)
        for rl in relevant_labels:
            relevant_mask |= labels == rl
        return z_what[relevant_mask], labels[relevant_mask]