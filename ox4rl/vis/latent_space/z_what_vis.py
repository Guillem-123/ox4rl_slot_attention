from ox4rl.dataset import get_label_list


import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from termcolor import colored


import os
import numpy as np


class ZWhatPlotter:

    NR_OF_DIMS = 2
    def __init__(self, cfg, folder, file_name, method="pca"):
        print("Initializing ZWhatPlotter with method", method)
        self.cfg = cfg
        self.folder = folder
        self.dim_red_path = f"{self.folder}/{file_name}"
        self.method = method
        self.DISPLAY_CENTROIDS = True
        self.COLORS = ['black', 'r', 'g', 'b', 'c', 'm', 'y', 'pink', 'purple', 'orange',
            'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
        self.edgecolors = False #True iff the ground truth labels and the predicted labels ''(Mixture of some greedy policy and NN) should be drawn in the same image
        self.annotate_wrong_preds = False # True iff the wrong predictions should be annotated in the visualization with the corresponding index of the z_what encoding (only works if edgecolors=True)

    def visualize_z_what(self, z_what, labels, pred_y, centroids, centroid_labels, relevant_labels):
        z_what_emb, centroid_emb, dim_name = self.perform_dimensionality_reduction(z_what, centroids) # either PCA or TSNE
        # check type for consistency before concatenating
        if isinstance(z_what,np.ndarray):
          z_what = torch.tensor(z_what, dtype=torch.float32)  
        
        print(z_what.size())
        print(labels.size())
        #train_all = torch.cat((z_what, labels.unsqueeze(1)), 1)
        train_all = torch.cat((z_what, labels), 1)
        
        # sort the indices
        sorted_indices = self.get_indices_for_each_label(train_all, relevant_labels)
        # get the label list
        label_list = get_label_list(self.cfg)
        # PLOT
        if self.edgecolors:
            self.edgecolor_visualization(len(z_what), relevant_labels, label_list, pred_y, centroid_labels, sorted_indices, z_what_emb, centroid_emb, dim_name)
        else:
            self.non_edgecolor_visualization(relevant_labels, label_list, pred_y, centroid_labels, sorted_indices, z_what_emb, centroid_emb, dim_name)

    def minimalistic_visualize_z_what(self, z_what, labels, pred_y, centroids, centroid_labels, relevant_labels):
        z_what_emb, centroid_emb, dim_name = self.perform_dimensionality_reduction(z_what, centroids)

        #import ipdb; ipdb.set_trace()
        if isinstance(z_what,np.ndarray):
          z_what = torch.tensor(z_what, dtype=torch.float32)
        
        #train_all_gt = torch.cat((z_what, labels.unsqueeze(1)), 1)
        train_all_gt = torch.cat((z_what, labels), 1)
        sorted_indices_gt = self.get_indices_for_each_label(train_all_gt, relevant_labels)
        # temporarily update dim_red_path
        tmp = self.dim_red_path
        self.dim_red_path = tmp + "_gt"
        self.minimalistic_visualization(sorted_indices_gt, z_what_emb, None, dim_name)
        self.dim_red_path = tmp

        # turn pred_y into a tensor via from_numpy
        pred_y = torch.from_numpy(pred_y)

        train_all_pred = torch.cat((z_what, pred_y.unsqueeze(1)), 1)
        sorted_indices_pred = self.get_indices_for_each_label(train_all_pred, np.arange(len(centroid_labels)))
        # temporarily update dim_red_path
        tmp = self.dim_red_path
        self.dim_red_path = tmp + "_pred"
        self.minimalistic_visualization(sorted_indices_pred, z_what_emb, centroid_emb, dim_name)
        self.dim_red_path = tmp
        



    @staticmethod
    def get_indices_for_each_label(train_all, relevant_labels):
        sorted_indices = []
        for i in relevant_labels:
            mask = train_all.T[-1] == i
            indices = torch.nonzero(mask)
            sorted_indices.append(indices)
        return sorted_indices

    def perform_dimensionality_reduction(self, z_what, centroids):
        # perform PCA or TSNE
        if self.method.lower() == "pca":
            print("Running PCA...")
            pca = PCA(n_components=self.NR_OF_DIMS)
            # check for correct type
            # before calling pca transformation
            if isinstance(z_what, torch.Tensor):
              z_what=z_what.numpy()
            z_what_emb = pca.fit_transform(z_what)
            
            centroid_emb = pca.transform(centroids)
            dim_name = "PCA"
        else:
            print("Running t-SNE...")
            print ("If too slow and GPU available, install cuml/MulticoreTSNE (requires conda)")
            tsne = TSNE(n_jobs=4, n_components=self.NR_OF_DIMS, verbose=True, perplexity=min(30, len(z_what)-1, len(centroids)-1))
            z_what_emb = tsne.fit_transform(z_what.numpy())
            centroid_emb = tsne.fit_transform(centroids)
            dim_name = "t-SNE"
        return z_what_emb, centroid_emb, dim_name
    
    def minimalistic_visualization(self, sorted_indices, z_what_emb, centroid_emb, dim_name):
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        axs.set_title("Labels", fontsize=20)
        axs.set_facecolor((81/255, 89/255, 99/255, 0.4))
        axs.set_xlabel(f"{dim_name} 1", fontsize=20)
        axs.set_ylabel(f"{dim_name} 2", fontsize=20)
        for i, idx in enumerate(sorted_indices):
            # dimension issue only if there is exactly one object of one kind
            colr = self.COLORS[i]
            axs.scatter(z_what_emb[:, 0][idx].squeeze(),
                               z_what_emb[:, 1][idx].squeeze(),
                               c=colr,
                               alpha=0.7)
        if self.DISPLAY_CENTROIDS and centroid_emb is not None:
            for i, c_emb in enumerate(centroid_emb):
                colr = self.COLORS[i]
                axs.scatter([c_emb[0]],
                                   [c_emb[1]],
                                   c=colr,
                                   edgecolors='black', s=100, linewidths=2)
        plt.tight_layout()
        self.save_plot(fig)

    def non_edgecolor_visualization(self, relevant_labels, label_list, y_pred, centroid_label, sorted_indices, z_what_emb, centroid_emb, dim_name):
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(15, 8)
        axs[0].set_title("Ground Truth Labels", fontsize=20)
        axs[1].set_title("Labels Following Clustering", fontsize=20)
        for ax in axs:
            ax.set_facecolor((81/255, 89/255, 99/255, 0.4))
            ax.set_xlabel(f"{dim_name} 1", fontsize=20)
            ax.set_ylabel(f"{dim_name} 2", fontsize=20)
        all_colors = []
        all_edge_colors = []
        for i, idx in enumerate(sorted_indices):
            # dimension issue only if there is exactly one object of one kind
            if torch.numel(idx) == 0:
                continue
            y_idx = y_pred[idx] if torch.numel(idx) > 1 else [[y_pred[idx]]]
            obj_name = relevant_labels[i]
            colr = self.COLORS[obj_name]
            edge_colors = [self.COLORS[centroid_label[assign[0]]] for assign in y_idx]
            all_edge_colors.extend(edge_colors)
            all_colors.append(colr)
            axs[0].scatter(z_what_emb[:, 0][idx].squeeze(),
                               z_what_emb[:, 1][idx].squeeze(),
                               c=colr,
                               label=label_list[obj_name],
                               alpha=0.7)
            axs[1].scatter(z_what_emb[:, 0][idx].squeeze(),
                               z_what_emb[:, 1][idx].squeeze(),
                               c=edge_colors,
                               alpha=0.7)
        if self.DISPLAY_CENTROIDS:
            for c_emb, cl in zip(centroid_emb, centroid_label):
                colr = self.COLORS[cl]
                axs[0].scatter([c_emb[0]],
                                   [c_emb[1]],
                                   c=colr,
                                   edgecolors='black', s=100, linewidths=2)
                axs[1].scatter([c_emb[0]],
                                   [c_emb[1]],
                                   c=colr,
                                   edgecolors='black', s=100, linewidths=2)

        axs[0].legend(prop={'size': 20})
        plt.tight_layout()

        self.save_plot(fig)

    def edgecolor_visualization(self, n, relevant_labels, label_list, y_pred, centroid_labels, sorted_indices, z_what_emb, centroid_emb, dim_name):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        ax.set_title(f"Labeled {self.method} of z_whats\n(Inner = GT label, Outer = pred label)")
        ax.set_facecolor((81/255, 89/255, 99/255, 0.4))
        ax.set_xlabel(f"{dim_name} 1", fontsize=20)
        ax.set_ylabel(f"{dim_name} 2", fontsize=20)
        n = min(n, 10000)
        for i, idx in enumerate(sorted_indices):
            if torch.numel(idx) == 0:
                continue
            y_idx = y_pred[idx] if torch.numel(idx) > 1 else [[y_pred[idx]]]
            obj_name = relevant_labels[i]
            colr = self.COLORS[obj_name]
            edge_colors = [self.COLORS[centroid_labels[assign[0]]] for assign in y_idx]
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:n],
                           z_what_emb[:, 1][idx].squeeze()[:n],
                           c=colr,
                           label=label_list[obj_name],
                           alpha=0.7, edgecolors=edge_colors, s=100, linewidths=2)

            if self.annotate_wrong_preds:
                # annotate all the points where edgecolors are different from colr
                for j, txt in enumerate(idx):
                    if edge_colors[j] != colr:
                        ax.annotate(txt.item(), (z_what_emb[:, 0][idx].squeeze()[j], z_what_emb[:, 1][idx].squeeze()[j]))
        if self.DISPLAY_CENTROIDS:
            for c_emb, cl in zip(centroid_emb, centroid_labels):
                colr = self.COLORS[cl]
                ax.scatter([c_emb[0]], [c_emb[1]],  c=colr, edgecolors='black', s=100, linewidths=2)

        ax.legend(prop={'size': 20})
        plt.tight_layout()

        self.save_plot(fig)

    def save_plot(self, fig):
        directory = f"{self.folder}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{self.dim_red_path}.svg")
        plt.savefig(f"{self.dim_red_path}.png")
        print(colored(f"Saved {self.method} images in {self.dim_red_path}", "blue"))
        plt.close(fig)


    def no_z_whats_plots(self):
        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(8, 15)
        axs[0].set_title("Ground Truth Labels", fontsize=20)
        axs[1].set_title("Labels Following Clustering", fontsize=20)
        s = "No z_what extracted\n      by the model"
        dim_name = "PCA" if self.method == "pca" else "t-SNE"
        for ax in axs:
            ax.set_xlabel(f"{dim_name} 1", fontsize=20)
            ax.set_ylabel(f"{dim_name} 2", fontsize=20)
            ax.text(0.03, 0.1, s, rotation=45, fontsize=45)
        plt.tight_layout()

        if not os.path.exists(f"{self.folder}"):
            os.makedirs(f"{self.folder}")
        plt.savefig(f"{self.dim_red_path}.svg")
        plt.savefig(f"{self.dim_red_path}.png")
        print(colored(f"Saved empty {self.method} images in {self.dim_red_path}", "red"))
        plt.close(fig)