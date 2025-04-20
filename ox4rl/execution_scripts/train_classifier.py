from ox4rl.latent_classification.z_what_classification import ZWhatClassifierCreator
import numpy as np
import os
import os.path as osp
from ox4rl.dataset.atari_labels import label_list_for, get_moving_indices
import pandas as pd
from ox4rl.dataset.atari_data_collector import AtariDataCollector
from ox4rl.vis.object_detection.classifier_vis import visualize_kmeans_classifier, visualize_x_means_classifier
from ox4rl.dataset_creation.create_latent_dataset import create_latent_dataset


def train_classifier(cfg):
    data_subset_mode = cfg.classifier.data_subset_mode
    dataset_mode = cfg.classifier.train_folder
    only_collect_first_image_of_consecutive_frames = cfg.classifier.one_image_per_sequence

    classifier_folder = osp.join(cfg.checkpointdir, cfg.exp_name)

    # collect the data
    latents_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], "latents", dataset_mode)
    if not os.path.exists(latents_path) or not os.listdir(latents_path):
        create_latent_dataset(cfg, dataset_mode=dataset_mode)
    else:
        print('Latent dataset already exists. Skipping creation.')

    z_whats, labels = AtariDataCollector.collect_z_what_data_reshaped(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    z_whats, labels = z_whats.cpu(), labels.cpu()

    if data_subset_mode == "relevant":
        relevant_labels = get_moving_indices(cfg.gamelist[0])
    elif data_subset_mode == "all":
        relevant_labels = np.arange(1, len(label_list_for(cfg.gamelist[0])))
    else:
        raise ValueError(f"Invalid data_subset_mode {data_subset_mode}")

    print("Training classifier on", len(z_whats), "z_what encodings")

    ### K-MEANS and NN CLASSIFIER based on K-MEANS ###
    kmeans, kmeans_clusters, kmeans_centers = create_k_means(cfg, z_whats, data_subset_mode, classifier_folder)
    # find out mapping of enumerated labels to actual labels (i.e. index in oc_atari labels)
    nn_neighbors_clf, centroids, centroid_labels = create_nn_classifier(cfg, kmeans, z_whats, labels, relevant_labels, data_subset_mode, classifier_folder)
    #save centroid_labels as a csv file
    df = pd.DataFrame(centroid_labels)
    kmeans_centroid_labels_path = osp.join(classifier_folder, f"z_what-classifier_{data_subset_mode}_kmeans_centroid_labels.csv")
    df.to_csv(kmeans_centroid_labels_path, header=False, index=True)

    ### X-MEANS ### you need to modify one line in the pyclustering package to run x_means https://github.com/annoviko/pyclustering/pull/697/files
    # xmeans_instance, xmeans_clusters, xmeans_centers = create_x_means(cfg, z_whats, data_subset_mode, classifier_folder_path)
    # visualize_x_means_classifier(cfg, dataset_mode, data_subset_mode, z_whats, labels, xmeans_centers, xmeans_instance, xmeans_clusters, relevant_labels, only_collect_first_image_of_consecutive_frames)


    if cfg.classifier.visualize:
        # select which classifier to use for visualization
        clf, _ , centers = kmeans, kmeans_clusters, kmeans_centers
        visualize_kmeans_classifier(cfg, dataset_mode, data_subset_mode, z_whats, labels, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames)


def create_x_means(cfg, train_x, data_subset_mode, classifier_folder):
    train_x = np.array(train_x)
    xmeans_instance = ZWhatClassifierCreator(cfg).create_x_means(train_x, kmax=3)
    save_classifier(cfg, xmeans_instance, "xmeans", data_subset_mode, classifier_folder)
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    return xmeans_instance, clusters, centers

def create_k_means(cfg, train_x, data_subset_mode, classifier_folder):
    # create a kmeans classifier
    k_means = ZWhatClassifierCreator(cfg).create_k_means(train_x, get_moving_indices(cfg.exp_name))
    save_classifier(cfg, k_means, "kmeans", data_subset_mode, classifier_folder)
    # collect the clusters and their centers for visualizations
    clusters = []
    for label in range(len(k_means.cluster_centers_)):
        clusters.append(np.where(k_means.labels_ == label)[0])
    centers = k_means.cluster_centers_
    return k_means, clusters, centers

def create_nn_classifier(cfg, kmeans, train_x, train_y, relevant_labels, data_subset_mode, classifier_folder):
    # create a nearest neighbor classifier
    nn_neighbors_clf, centroids, centroid_labels = ZWhatClassifierCreator(cfg).nn_clf_based_on_k_means_centroids(kmeans, train_x, train_y, relevant_labels)
    save_classifier(cfg, nn_neighbors_clf, "nn", data_subset_mode, classifier_folder)
    return nn_neighbors_clf, centroids, centroid_labels

def save_classifier(cfg, clf, clf_name, data_subset_mode, classifier_folder):
    ZWhatClassifierCreator(cfg).save_classifier(
        clf=clf,
        clf_name=f"{clf_name}",
        folder=classifier_folder,
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












    
