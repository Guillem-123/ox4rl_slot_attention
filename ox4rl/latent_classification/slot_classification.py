import os
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
# import x-means from pyclustering
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import ipdb

import joblib
import numpy as np

class SlotClassifierCreator:

    few_shot_values = [1, 4, 16, 64]
    N_NEIGHBORS = 24

    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(f'{self.cfg.logdir}/{self.cfg.exp_name}', exist_ok=True)
        os.makedirs(f'classifiers', exist_ok=True)

    def create_ridge_classifiers(self, relevant_labels, train_x, train_y):
        # separate the data by class label
        z_what_by_class_label = {rl: train_x[train_y == rl] for rl in relevant_labels}
        labels_by_class_label = {rl: train_y[train_y == rl] for rl in relevant_labels}

        classifiers = {}
        for training_objects_per_class in ZWhatClassifierCreator.few_shot_values:
            current_train_sample = torch.cat([z_what_by_class_label[rl][:training_objects_per_class] for rl in relevant_labels])
            current_train_labels = torch.cat([labels_by_class_label[rl][:training_objects_per_class] for rl in relevant_labels])
            classifiers[training_objects_per_class] = self.create_ridge_classifier(current_train_sample, current_train_labels)
        return classifiers
    
    # def create_ridge_classifier(self, slot_representations, labels):
    #     """ Train a Ridge classifier using slot-based representations """
    #     if not isinstance(slot_representations, torch.Tensor) or not isinstance(labels, torch.Tensor):
    #         raise TypeError("Inputs must be PyTorch tensors.")

    #     num_samples, num_slots, slot_dim = slot_representations.shape  # (B, num_slots, slot_dim)

    #     # Ensure labels have shape (B,)
    #     if labels.dim() == 3:  
    #         labels = labels.squeeze(1)  # Remove extra dimensions if they exist
    #     elif labels.dim() > 2:
    #         raise ValueError(f"Unexpected shape for labels: {labels.shape}")


    #     # Flatten slots: (B, num_slots, slot_dim) → (B * num_slots, slot_dim)
    #     train_x = slot_representations.view(-1, slot_dim).numpy()

    #     # Correctly expand labels to match slots: (B,) → (B * num_slots,)
    #     train_y = labels.unsqueeze(1).expand(-1, num_slots).reshape(-1).numpy()

    #     # Initialize and train Ridge Classifier
    #     clf = RidgeClassifier()
    #     clf.fit(train_x, train_y)

    #     return clf

 
    def create_ridge_classifier(self, slot_representations, labels):
      """Train a Ridge classifier using slot-based representations."""
      if not isinstance(slot_representations, np.ndarray) or not isinstance(labels, np.ndarray):
          raise TypeError("Inputs must be NumPy arrays.")

      # Reshape labels to a 1D array if necessary
      if labels.ndim > 1 and labels.shape[1] == 1:
          labels = labels.ravel()

      # Initialize and train Ridge Classifier
      clf = RidgeClassifier()
      clf.fit(slot_representations, labels)

      return clf




    def create_k_means(self, slot_representations, relevant_labels):
        """ Train a K-Means clustering model on slot representations """

        # # Ensure input is a tensor or NumPy array
        # if not isinstance(slot_representations, (torch.Tensor, np.ndarray)):
        #     raise TypeError("slot_representations must be a PyTorch tensor or NumPy array.")

        # Convert PyTorch tensor to NumPy if needed (sklearn KMeans requires NumPy)
        if isinstance(slot_representations, torch.Tensor):
            slot_representations = slot_representations.numpy()

        # Ensure slot_representations is already in shape (N, slot_dim)
        if len(slot_representations.shape) != 2:
            raise ValueError(f"Expected slot_representations shape (N, slot_dim), but got {slot_representations.shape}")

        # Create and fit K-Means model
        k_means = KMeans(n_clusters=len(relevant_labels))
        k_means.fit(slot_representations)

        return k_means

    def save_classifier(cfg, clf, clf_name, data_subset_mode):
        print(f"Saving classifier: {clf_name}")

        # Use `osp.join` to ensure cross-platform compatibility
        # folder = os.path.join(cfg.dataset_roots.ATARI, "output", "checkpoints", "final", "pong")
        folder = "/content/content/ox4rl/ox4rl/output/checkpoints/final/pong"


        # Ensure the directory exists
        os.makedirs(folder, exist_ok=True)

        # Define filename properly
        filename = f"slot-classifier_{data_subset_mode}_{clf_name}.joblib.pkl"
        path = os.path.join(folder, filename)

        # Save classifier
        joblib.dump(clf, path)
        print(f"Classifier saved at: {path}")



    # high level: essentially assign semantic labels to k_means centroids instead of just enumerating the clusters

    def nn_clf_based_on_k_means_centroids(self, k_means, train_x, train_y, relevant_labels):
        X = train_x
        y = train_y.squeeze()  # Ensure y is 1D

        # Get K-means centroids
        centroids = k_means.cluster_centers_

        # Debug: Check label consistency
        print(f"Relevant labels: {relevant_labels}")
        print(f"Unique train_y labels: {np.unique(y)}")  # Convert to NumPy to check values

        # Nearest neighbors search
        n_neighbors = min(self.N_NEIGHBORS, len(X))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        _, z_w_idx = nn.kneighbors(centroids)

        centroid_label = []
        for cent, nei in zip(centroids, z_w_idx):
            count = {rl: 0 for rl in relevant_labels}  # Initialize count dict
            added = False

            for i in range(n_neighbors):
                nei_label = y[nei[i]].item()  # Extract label

                if nei_label not in count:  # Prevent KeyError
                    print(f"Warning: Label {nei_label} not found in relevant_labels. Skipping...")
                    continue  # Skip invalid labels

                count[nei_label] += 1

                threshold = (6.0 / (i + 1)) if nei_label in centroid_label else (3.0 / (i + 1))
                if count[nei_label] > threshold:
                    centroid_label.append(nei_label)
                    added = True
                    break

            if not added:
                leftover_labels = [i for i in relevant_labels if i not in centroid_label]

                if leftover_labels:
                    centroid_label.append(leftover_labels[0])
                else:
                    print("Warning: No leftover labels available, using the most frequent label.")
                    most_frequent_label = max(count, key=count.get, default=relevant_labels[0])
                    centroid_label.append(most_frequent_label)

        # Ensure labels match number of centroids
        assert len(centroid_label) == len(centroids), f"Mismatch: {len(centroid_label)} labels for {len(centroids)} centroids"

        # Train a 1-NN classifier on centroids
        nn_class = KNeighborsClassifier(n_neighbors=1)
        nn_class.fit(centroids, centroid_label)  # Convert to NumPy array for safety

        return nn_class, centroids, centroid_label



    def create_x_means(self, sample, kmax=20):
        amount_initial_centers = 2
        initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers).initialize()
        # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
        # number of clusters that can be allocated is 20.
        xmeans_instance = xmeans(sample, initial_centers, kmax=kmax)
        xmeans_instance.process()
        return xmeans_instance

    #def create_gaussian_mixture(self, sample, n_components):
    #    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    #    gmm.fit(sample)
    #    labels = gmm.predict(sample)
    #    return gmm  


    