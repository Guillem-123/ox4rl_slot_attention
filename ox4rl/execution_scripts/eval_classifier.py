import pandas as pd
import numpy as np
import os
import os.path as osp
from ox4rl.dataset.atari_data_collector import AtariDataCollector
from ox4rl.training.checkpointing.loading import load_classifier
from sklearn.metrics import classification_report
from ox4rl.dataset_creation.create_latent_dataset import create_latent_dataset
from ox4rl.dataset.atari_labels import label_list_for, get_moving_indices
from ox4rl.vis.object_detection.classifier_vis import visualize_nn_classifier

def eval_classifier(cfg):
    dataset_mode = cfg.classifier.test_folder
    data_subset_mode = cfg.classifier.data_subset_mode
    clf_name = cfg.classifier.clf_name
    only_collect_first_image_of_consecutive_frames = cfg.classifier.one_image_per_sequence

    # Check if latent dataset already exists to avoid redundant creation
    latents_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], "latents", dataset_mode)
    if not os.path.exists(latents_path) or not os.listdir(latents_path):
        create_latent_dataset(cfg, dataset_mode)
    else:
        print('Latent dataset already exists. Skipping creation. Warning: Make sure that the latent dataset is up-to-date.')

    classifier_folder = osp.join(cfg.checkpointdir, cfg.exp_name)
    eval_classifier_folder = osp.join(cfg.evaldir, cfg.exp_name)

    clf, centroid_labels = load_classifier(
        folder_path=classifier_folder,
        clf_name=clf_name,
        data_subset_mode=data_subset_mode
    )
    
    # Collect the data
    x, y = AtariDataCollector.collect_z_what_data_reshaped(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    x, y = x.cpu(), y.cpu() # put all data onto cpu

    print("Evaluating classifier on", len(x), "z_what encodings")

    # Evaluate the classifier
    pred_y = clf.predict(x)
    if centroid_labels is not None:
        pred_y = [centroid_labels[pred_y_i] for pred_y_i in pred_y]
 
    # Generate metrics and print classification report
    metric_dict = classification_report(y, pred_y, output_dict=True)
    for k, v in metric_dict.items():
        print(f"{k}: {v}")
    
    # Save metric_dict as csv
    df = pd.DataFrame(metric_dict)
    eval_classifier_filepath = osp.join(eval_classifier_folder, "eval_classifier.csv")
    df.to_csv(eval_classifier_filepath, header=True, index=True)

    if cfg.classifier.visualize:
        if data_subset_mode == "relevant":
            relevant_labels = get_moving_indices(cfg.gamelist[0])
        elif data_subset_mode == "all":
            relevant_labels = np.arange(1, len(label_list_for(cfg.gamelist[0])))
        else:
            raise ValueError(f"Invalid data_subset_mode {data_subset_mode}")


        # Make some transformations for ZWhatPlotter to work (within visualize_nn_classifier)
        centers = clf._fit_X
        centroid_labels = clf.classes_[clf._y] # Map the enumerated labels back to the ocatari based numbering of class labels
        # sort centers and centroids by the same way clf._y would be sorted: zip with clf._y and sort by it
        _ , centers = zip(*sorted(zip(clf._y, centers), key=lambda x: x[0]))
        _ , centroid_labels = zip(*sorted(zip(clf._y, centroid_labels), key=lambda x: x[0]))


        visualize_nn_classifier(cfg, dataset_mode, data_subset_mode, x, y, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames)


if __name__ == "__main__":
    import argparse
    from ox4rl.utils.load_config import get_config_v2
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", metavar="FILE", help="path to config file", type=str)
    config_path = parser.parse_args().config_file
    cfg = get_config_v2(config_path)
    eval_classifier(cfg)

