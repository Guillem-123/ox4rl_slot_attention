import os
from ox4rl.vis.object_detection.space_vis import class_label_bb
from ox4rl.models.space.space import Space
import torch
from PIL import Image
import torchvision.transforms as transforms
from ox4rl.training.checkpointing.loading import load_model
from ox4rl.utils.load_config import get_config_v2
import argparse
import joblib
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans
from ox4rl.latent_classification.z_what_classification import ZWhatClassifierCreator

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)
os.makedirs("input", exist_ok=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate class-specific bounding boxes")
parser.add_argument("--config", type=str, default="../configs/my_atari_pong.yaml", 
                   help="Path to config file")
parser.add_argument("--image_path", type=str, default="input/0000_0.png",
                   help="Path to input image")
parser.add_argument("--output_path", type=str, default="output/0000_0_classed_label_bb.png",
                   help="Path to save output image")
parser.add_argument("--classifier_path", type=str, 
                   help="Path to pre-trained classifier (optional)")
parser.add_argument("--use_kmeans", action="store_true",
                   help="Use K-means clustering instead of a pre-trained classifier")
args = parser.parse_args()

# Load configuration and model
config_path = args.config
cfg = get_config_v2(config_path)
model, _, _, _, _ = load_model(cfg, mode="eval")

# Load the test image
image_path = args.image_path
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    exit()
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)

# Get or create classifier
classifier = None
if args.classifier_path and os.path.exists(args.classifier_path):
    # Load pre-trained classifier
    print(f"Loading classifier from {args.classifier_path}")
    classifier = joblib.load(args.classifier_path)
elif args.use_kmeans:
    # Create a simple KMeans classifier with 4 clusters
    print("No classifier provided, creating a simple K-means classifier")
    classifier = KMeans(n_clusters=4, random_state=42)
    
    # Run model once to get z_what embeddings for a sample
    with torch.no_grad():
        _, log = model(image_tensor, 100000000)
        z_what = log.get("z_what", None)
        
    if z_what is not None:
        # Train KMeans on this sample (not ideal but works as demo)
        # Reshape z_what to 2D array for KMeans
        z_what_np = z_what.cpu().detach().reshape(-1, z_what.shape[-1]).numpy().astype('float64')
        classifier.fit(z_what_np)
        print("Trained K-means classifier on sample image embeddings")
    else:
        print("Warning: Model did not output z_what, classifier won't work")
        classifier = None
else:
    print("No classifier provided and --use_kmeans not specified. Running without classification.")

# Call the class_label_bb function to label bounding boxes according to classes
output_path = args.output_path
class_label_bb(model, image_tensor, path=output_path, device="cpu", classifier=classifier, game_name="Pong")
print(f"Labeled image with class-specific bounding boxes saved to: {output_path}")