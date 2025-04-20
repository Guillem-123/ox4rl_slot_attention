
import matplotlib
import numpy as np
import os
import os.path as osp
import torch
from PIL import Image
from ox4rl.dataset.atari_data_collector import AtariDataCollector
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib



from ox4rl.vis.latent_space.z_what_vis import ZWhatPlotter


def visualize_x_means_classifier(cfg, dataset_mode, data_subset_mode, z_whats, labels, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames):
    classifier_eval_path = osp.join(cfg.evaldir, cfg.exp_name)
    base_folder_path = osp.join(classifier_eval_path, "classifier_visualization")
    os.makedirs(base_folder_path, exist_ok=True)

    # Visualize clustering results
    dim_red_file_name = f"{clf.__class__.__name__}_{data_subset_mode}_clusters_PCA_{dataset_mode}"
    z_whats = np.ascontiguousarray(z_whats)
    y_pred = clf.predict(z_whats)
    z_whats,labels=torch.from_numpy(z_whats),torch.from_numpy(labels)
    ZWhatPlotter(cfg, base_folder_path, dim_red_file_name).minimalistic_visualize_z_what(z_whats, labels, y_pred, centers, centroid_labels, relevant_labels)



def visualize_kmeans_classifier(cfg, dataset_mode, data_subset_mode, z_whats, labels, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames):
    classifier_eval_path = osp.join(cfg.evaldir, cfg.exp_name)
    base_folder_path = osp.join(classifier_eval_path, "classifier_visualization")
    os.makedirs(base_folder_path, exist_ok=True)

    # Visualize clustering results
    dim_red_file_name = f"{clf.__class__.__name__}_{data_subset_mode}_clusters_PCA_{dataset_mode}"
    y_pred = clf.predict(z_whats)
    z_whats,labels=torch.from_numpy(z_whats),torch.from_numpy(labels)
    ZWhatPlotter(cfg, base_folder_path, dim_red_file_name).visualize_z_what(z_whats, labels, y_pred, centers, centroid_labels, relevant_labels)

    # Visualize false predictions
    visualize_false_predictions_slot(cfg, dataset_mode, data_subset_mode, labels, centroid_labels, y_pred, only_collect_first_image_of_consecutive_frames)

    # Visualize the selected bbox for each cluster
    images = AtariDataCollector.collect_images(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    #pred_boxes = AtariDataCollector.collect_pred_boxes(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    slot_representations,_ = AtariDataCollector.collect_slot_attention_data(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    slot_masks= AtariDataCollector.collect_slot_masks(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    #z_whats, labels = AtariDataCollector.collect_z_what_data(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    create_cluster_folders_slot(clf, images, slot_masks, slot_representations, labels, base_folder_path)
    create_one_grid_image_for_each_cluster(base_folder_path)


def visualize_nn_classifier(cfg, dataset_mode, data_subset_mode, z_whats, labels, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames):
    classifier_eval_path = osp.join(cfg.evaldir, cfg.exp_name)
    base_folder_path = osp.join(classifier_eval_path, "classifier_visualization")
    os.makedirs(base_folder_path, exist_ok=True)

    # Visualize clustering results
    dim_red_file_name = f"{clf.__class__.__name__}_{data_subset_mode}_clusters_PCA_{dataset_mode}"
    y_pred = clf.predict(z_whats)

    # map the y_pred from classes_ to clf_y; y_pred is in 1,2,4 etc. and clf_y is in 0,1,2 etc.
    dict_map = {key: value for key, value in zip(clf.classes_[clf._y], clf._y)}
    y_pred = [dict_map[y_pred_i] for y_pred_i in y_pred]
    y_pred = np.array(y_pred)

    ZWhatPlotter(cfg, base_folder_path, dim_red_file_name).visualize_z_what(z_whats, labels, y_pred, centers, centroid_labels, relevant_labels)

def visualize_false_predictions(cfg, dataset_mode, data_subset_mode, labels, centroid_labels, y_pred, only_collect_first_image_of_consecutive_frames):
     # Visualize false predictions
    images = AtariDataCollector.collect_images(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    pred_boxes = AtariDataCollector.collect_pred_boxes(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    cat_pred_boxes = torch.cat(pred_boxes, dim=0)
    
    image_ref_indexes = []
    for i in range(len(pred_boxes)):
        for j in range(len(pred_boxes[i])):
            image_ref_indexes.append(i)

    classifier_eval_path = osp.join(cfg.evaldir, cfg.exp_name)
    base_path = osp.join(classifier_eval_path, "classifier_visualization", "wrong_predictions")
    os.makedirs(osp.join(base_path), exist_ok=True)
    y_pred = [centroid_labels[y_pred_i] for y_pred_i in y_pred]

    for i, (y_pred_i, y_gt_i, pred_box_i, image_ref_index) in enumerate(zip(y_pred, labels, cat_pred_boxes, image_ref_indexes)):
        # store examples of all wrong predictions
        if y_pred_i != y_gt_i.item():
            image = images[image_ref_index]
            img = get_bbox_patch_of_image(image, pred_box_i)
            img.save(osp.join(base_path, f"pred_{y_pred_i}_gt_{y_gt_i}_{i}.png"))


def visualize_false_predictions_slot(cfg, dataset_mode, data_subset_mode, labels, centroid_labels, y_pred, only_collect_first_image_of_consecutive_frames):
    # Collect images and slot masks
    images = AtariDataCollector.collect_images(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    slot_masks = AtariDataCollector.collect_slot_masks(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    classifier_eval_path = osp.join(cfg.evaldir, cfg.exp_name)
    base_path = osp.join(classifier_eval_path, "classifier_visualization", "wrong_predictions")
    os.makedirs(base_path, exist_ok=True)

    y_pred = [centroid_labels[y_pred_i] for y_pred_i in y_pred]  # Convert predicted labels to actual class names

    for i, (y_pred_i, y_gt_i, image, masks) in enumerate(zip(y_pred, labels, images, slot_masks)):
        if y_pred_i != y_gt_i.item():  # Only visualize wrong predictions
            for j, mask in enumerate(masks):  
                masked_img = apply_slot_mask(image, mask)  # ✅ Apply slot mask to the image

                # Save incorrectly classified images with masks
                masked_img.save(osp.join(base_path, f"pred_{y_pred_i}_gt_{y_gt_i}_{i}_{j}.png"))


def get_bbox_patch_of_image(img, pred_box):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    pred_box = pred_box.detach().cpu().numpy()
    pred_box = pred_box*128
    pred_box = np.clip(pred_box, 0, 127) #protect against negative values
    pred_box = pred_box.astype(np.uint8)  
    img = img[pred_box[0]:pred_box[1], pred_box[2]:pred_box[3]]
    img = Image.fromarray(img)
    return img

#def perform_dimensionality_reduction(z_what, centroids,):
#    # perform PCA or TSNE
#    pca = PCA(n_components=2)
#    z_what_emb = pca.fit_transform(z_what)
#    centroid_emb = pca.transform(np.array(centroids))
#    dim_name = "PCA"
#    return z_what_emb, centroid_emb, dim_name

def create_cluster_folders(clf, images, pred_boxes, z_whats, labels, base_path):
    for i, (imgs, pred_boxes, z_what, label) in enumerate(zip(images, pred_boxes, z_whats, labels)):
        z_what = z_what.cpu()
        pred_labels = clf.predict(z_what)
        # cut out the relevant bbox for each pred_box
        for j, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            # cut out the relevant bbox
            img = get_bbox_patch_of_image(imgs, pred_box)
            # save the bbox
            folder = f"cluster_{pred_label}"
            os.makedirs(osp.join(base_path, folder), exist_ok=True)
            img.save(osp.join(base_path, folder, f"{i}_{j}.png"))
        if i == 100:
            break

import torch.nn.functional as F
def apply_slot_mask(image, mask):
    """
    Applies a slot attention mask to an image.

    Args:
    - image (torch.Tensor): Shape (C, H, W).
    - mask (torch.Tensor): Shape (num_slots, H', W').

    Returns:
    - PIL Image with the applied mask.
    """
    # print("printing mask shape")
    # print(mask.shape)
    # print("----------------------")
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    mask = mask.cpu()  # Convert to CPU tensor

    # Handle multiple slot masks
    if mask.ndim == 3:  
        mask = mask[0]  # Select first slot OR use mask.mean(0) for averaging
    print("Mask shape before unsqueezing")
    print(mask.shape)
    mask = mask.reshape(1, 1, *mask.shape)  # Shape becomes [1, 1, 32, 32,1] # I ADDED THIS LINE
   
    #mask = mask.unsqueeze(-1)  # Reshape to (N, C, H, W)

    print("Mask shape after unsqueezing")
    print(mask.shape)
    mask = F.interpolate(mask, size=(128, 128), mode="bilinear", align_corners=False)  # Resize
    mask = mask.squeeze().numpy()  # Remove extra dimensions

    # Apply mask
    masked_image = image * np.expand_dims(mask, axis=-1)  
    masked_image = (masked_image * 255).astype(np.uint8)

    return Image.fromarray(masked_image)

def create_cluster_folders_slot(clf, images, slot_masks, slot_representations, labels, base_path):
    for i, (img, masks, slots, label) in enumerate(zip(images, slot_masks, slot_representations, labels)):
        
        #slots=slots.cpu()
        pred_labels = clf.predict(slots.reshape(1,-1))
        
        for j, (mask, pred_label) in enumerate(zip(masks, pred_labels)):
            masked_img = apply_slot_mask(img, mask)  # New function to visualize slots
            folder = f"cluster_{pred_label}"
            os.makedirs(osp.join(base_path, folder), exist_ok=True)
            masked_img.save(osp.join(base_path, folder, f"{i}_{j}.png"))
        
        if i == 100:  # Limit the number of images
            break



def create_one_grid_image_for_each_cluster(base_path):
    # get all cluster folders: check whether is directory and starts with "cluster_"
    cluster_folders = [osp.join(base_path, folder) for folder in os.listdir(base_path) if osp.isdir(osp.join(base_path, folder)) and folder.startswith("cluster_")]
    # open each folder iteratively
    for folder in cluster_folders:
        x_size, y_size = (20,20)
        # create grid image containing each image in the folder
        image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        if not image_files:
            print(f"No images found in {folder}")
            continue

        # Create a new blank image to accommodate all the small images
        grid_size = (int(np.sqrt(len(image_files))), int(np.ceil(len(image_files) / np.sqrt(len(image_files)))))
        grid_image = Image.new('RGB', (grid_size[0] * x_size, grid_size[1] * y_size))

        # Iterate through each image and paste it into the grid
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(folder, image_file)
            img = Image.open(image_path)

            # Resize the image to fit in the grid (adjust as needed)
            img = img.resize((x_size, y_size), Image.LANCZOS)

            # Calculate the position to paste the image
            row = i // grid_size[0]
            col = i % grid_size[0]

            # Paste the resized image into the grid
            grid_image.paste(img, (col * x_size,  row * y_size))

        # Save the grid image for the current cluster
        base_folder, cluster = folder.rsplit("/", 1)
        base_folder = base_folder + "/cluster_grid_images"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        grid_image.save(os.path.join(base_folder,  f"{cluster}_grid.png"))

def visualize_ridge_classifier(slot_representations, labels):
    if not isinstance(slot_representations, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Inputs must be PyTorch tensors.")

    train_x = slot_representations.numpy()
    train_y = labels.numpy().ravel()

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)

    pca = PCA(n_components=2)
    train_x_pca = pca.fit_transform(train_x_scaled)

    clf = RidgeClassifier()
    clf.fit(train_x_pca, train_y)

    x_min, x_max = train_x_pca[:, 0].min() - 1, train_x_pca[:, 0].max() + 1
    y_min, y_max = train_x_pca[:, 1].min() - 1, train_x_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    scatter = plt.scatter(train_x_pca[:, 0], train_x_pca[:, 1], c=train_y, edgecolor="k", cmap=plt.cm.coolwarm)

    # ✅ Fix legend handling
    handles, _ = scatter.legend_elements()
    labels = [str(lbl) for lbl in np.unique(train_y)]
    plt.legend(handles, labels, title="Classes")

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Ridge Classifier Decision Boundary")
    plt.savefig("ridge_classifier.png")  # Save plot to file
    plt.show()



