from ox4rl.latent_classification.slot_classification import SlotClassifierCreator
import numpy as np
import os
import os.path as osp
from ox4rl.dataset.atari_labels import label_list_for, get_moving_indices
import pandas as pd
from ox4rl.dataset.atari_data_collector import AtariDataCollector
#from ox4rl.eval.eval_classification.classifier_vis import visualize_classifier
from ox4rl.vis.object_detection.classifier_vis import visualize_kmeans_classifier, visualize_x_means_classifier, visualize_ridge_classifier
from ox4rl.dataset_creation.create_latent_dataset_slot import create_latent_dataset
from ox4rl.dataset.atari_dataset import Atari_Z_What
import joblib
#import ipdb

def train_classifier(cfg):
    data_subset_mode = cfg.classifier.data_subset_mode
    dataset_mode = cfg.classifier.train_folder
    only_collect_first_image_of_consecutive_frames = cfg.classifier.one_image_per_sequence
    #print(cfg) cfg.model==slot

    # TODO: get the slots, labels in the right way
    # cant do that need to use dataloader
    # imgs, slots, labels = Atari_Z_What(cfg, dataset_mode, return_keys=["imgs", "slot_latents", "slot_latents_labels"])

    base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], "latents_slot", dataset_mode)
    print(base_path)
    if not os.path.exists(base_path) or not os.listdir(base_path):
        create_latent_dataset(cfg, dataset_mode=dataset_mode)
    else:
        print('Latent dataset already exists. Skipping creation.')

    slots_latents, slot_latents_labels = AtariDataCollector.collect_slot_attention_data(cfg,dataset_mode,data_subset_mode)

    # Load Slot Attention embeddings
    # slot_latents, labels = AtariDataCollector.collect_slot_attention_data(cfg, dataset_mode, data_subset_mode)
    # slots, labels = slots.cpu(), labels.cpu()

    # Ensure slots have shape (N, slot_dim) before passing to classifiers
    # if len(slots_latents.shape) == 3:  # (B, num_slots, slot_dim)
    #     slots_latents = slots_latents.view(-1, slots_latents.shape[-1])  # Flatten to (N, slot_dim)
    #     slot_latents_labels = slot_latents_labels.unsqueeze(1).expand(-1, slots_latents.shape[0] // slot_latents_labels.shape[0]).reshape(-1)  # Expand labels

    if data_subset_mode == "relevant":
        relevant_labels = get_moving_indices(cfg.gamelist[0])
    elif data_subset_mode == "all":
        relevant_labels = np.arange(1, len(label_list_for(cfg.gamelist[0])))
    else:
        raise ValueError(f"Invalid data_subset_mode {data_subset_mode}")

    print(f"Training classifier on {len(slots_latents)} Slot Attention embeddings")

    ### Train Slot-Based Ridge Classifier ###
    ridge_clf = create_ridge_classifier(cfg, slots_latents, slot_latents_labels, data_subset_mode)

    ### Train Slot-Based K-Means ###
    kmeans, kmeans_clusters, kmeans_centers = create_k_means(cfg, slots_latents, data_subset_mode)# TODO checkout bug with relevant_labels and data_subset_mode, same in Nils' implementation
    #ipdb.set_trace()

    nn_neighbors_clf, centroids, centroid_labels = create_nn_classifier(cfg, kmeans, slots_latents, slot_latents_labels, relevant_labels, data_subset_mode)

    print("Centroid labels:")
    print(centroid_labels)
    #save centroid_labels as a csv file
    df = pd.DataFrame(centroid_labels)
    #df.to_csv(f"/ox4rl/ox4rl/output/checkpoints/final/pong/slot-classifier_{data_subset_mode}_kmeans_centroid_labels.csv", header=False, index=True)
    df.to_csv(fr"/content/content/ox4rl/ox4rl/output/checkpoints/final/pongslot-classifier_{data_subset_mode}_kmeans_centroid_labels.csv", header=False, index=True)

    if cfg.classifier.visualize:
        # select which classifier to use for visualization
        visualize_ridge_classifier(slots_latents, slot_latents_labels)
        clf, clusters, centers = kmeans, kmeans_clusters, kmeans_centers
        visualize_x_means_classifier(cfg, dataset_mode, data_subset_mode, slots_latents, slot_latents_labels, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames)
        visualize_kmeans_classifier(cfg, dataset_mode, data_subset_mode, slots_latents, slot_latents_labels, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames)

        

    
def create_x_means(cfg, train_x, data_subset_mode):
    
    xmeans_instance = SlotClassifierCreator(cfg).create_x_means(train_x, kmax=3)
    save_classifier(cfg, xmeans_instance, "xmeans", data_subset_mode)
    
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    return xmeans_instance, clusters, centers

def create_k_means(cfg, train_x, data_subset_mode):
    # TODO check if those labels are correctly taken and actually relevant
    relevant_labels = get_moving_indices(cfg.gamelist[0]) if data_subset_mode == "relevant" else np.arange(1, len(label_list_for(cfg.gamelist[0])))

    # create a kmeans classifier
    k_means = SlotClassifierCreator(cfg).create_k_means(train_x, relevant_labels)
    print(f"Printing data_subset_mode:{data_subset_mode} ")
    save_classifier(cfg, k_means, "kmeans", data_subset_mode)

    # collect the clusters and their centers for visualizations
    clusters = [np.where(k_means.labels_ == label)[0] for label in range(len(k_means.cluster_centers_))]
    centers = k_means.cluster_centers_
    return k_means, clusters, centers

def create_nn_classifier(cfg, kmeans, train_x, train_y, relevant_labels, data_subset_mode):
    # create a nearest neighbor classifier
    nn_neighbors_clf, centroids, centroid_labels = SlotClassifierCreator(cfg).nn_clf_based_on_k_means_centroids(kmeans, train_x, train_y, relevant_labels)
    save_classifier(cfg, nn_neighbors_clf, "nn", data_subset_mode)
    return nn_neighbors_clf, centroids, centroid_labels

def create_ridge_classifier(cfg, slots, slot_labels,data_subset_mode):
    ridge_classifier_clf = SlotClassifierCreator(cfg).create_ridge_classifier(slots, slot_labels)
    save_classifier(cfg, ridge_classifier_clf, "ridge_classifier", data_subset_mode)
    return ridge_classifier_clf


def save_classifier(cfg, clf, clf_name, data_subset_mode):
    SlotClassifierCreator(cfg).save_classifier(
        clf=clf,
        clf_name=f"{clf_name}",
        data_subset_mode=data_subset_mode
    )

if __name__ == "__main__":
    import argparse
    from ox4rl.utils.load_config import get_config_v2
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    config_path = parser.parse_args().config_file
    cfg = get_config_v2(config_path)
    train_classifier(cfg)