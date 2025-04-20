import os
import joblib
import argparse
import torch
import numpy as np
from ox4rl.utils.load_config import get_config_v2
from ox4rl.latent_classification.z_what_classification import ZWhatClassifierCreator
from ox4rl.dataset.atari_data_collector import AtariDataCollector

# Create a script to train and save the classifier
parser = argparse.ArgumentParser(description="Train and save a z_what classifier")
parser.add_argument("--config", type=str, default="ox4rl/configs/my_atari_pong.yaml", 
                   help="Path to config file")
parser.add_argument("--output_dir", type=str, default="classifiers", 
                   help="Directory to save the classifier")
parser.add_argument("--dataset_mode", type=str, default="train", 
                   choices=["train", "test", "validation"],
                   help="Which dataset split to use for training")
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load config
cfg = get_config_v2(args.config)

# Create classifier creator
clf_creator = ZWhatClassifierCreator(cfg)

try:
    print(f"Collecting z_what data from {args.dataset_mode} dataset...")
    z_whats, labels = AtariDataCollector.collect_z_what_data(
        cfg, args.dataset_mode, "all", 
        only_collect_first_image_of_consecutive_frames=True
    )
    
    if z_whats is not None and len(z_whats) > 0:
        print(f"Collected {len(z_whats)} z_what embeddings with {len(torch.unique(labels))} unique labels")
        
        # Find unique labels
        unique_labels = torch.unique(labels).tolist()
        print(f"Unique labels: {unique_labels}")
        
        # Train ridge classifier
        print("Training Ridge classifier...")
        ridge_clf = clf_creator.create_ridge_classifier(z_whats, labels)
        
        # Save the classifier
        output_path = os.path.join(args.output_dir, "z_what_ridge_classifier.joblib")
        joblib.dump(ridge_clf, output_path)
        print(f"Ridge classifier saved to {output_path}")
        
        # Also train the few-shot classifiers (very effective!)
        print("Training few-shot classifiers...")
        few_shot_clfs = clf_creator.create_ridge_classifiers(unique_labels, z_whats, labels)
        
        # Save the few-shot classifiers
        for n_shots, clf in few_shot_clfs.items():
            output_path = os.path.join(args.output_dir, f"z_what_ridge_{n_shots}shot_classifier.joblib")
            joblib.dump(clf, output_path)
            print(f"{n_shots}-shot classifier saved to {output_path}")
        
    else:
        print("No z_what data found.")
        
except Exception as e:
    print(f"Error training classifier: {e}")