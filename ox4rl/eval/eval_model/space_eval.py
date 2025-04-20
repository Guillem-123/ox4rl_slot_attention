from ox4rl.eval.latent_eval.clustering_eval import ClusteringEval
from ox4rl.eval.eval_model.ap_and_acc_eval import ApAndAccEval
from ox4rl.eval.eval_model.ap_and_acc_eval import THRESHOLDS
import numpy as np
import torch
import torch.nn.functional as F
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import pprint
from ox4rl.dataset_creation.create_latent_dataset import create_latent_dataset
from ox4rl.dataset.atari_dataset import Atari_Z_What 
from ox4rl.dataset import get_dataloader  
from ox4rl.training.metric_logger import MetricLogger
from ox4rl.dataset.atari_labels import get_moving_indices, label_list_for
from dataclasses import dataclass

@dataclass
class EvalConfig:
    """
    The EvalConfig dataclass defines the configuration settings required for the evaluation pipeline.
    
    Attributes:
        logdir (str): The directory where logs are stored.
        exp_name (str): The name of the experiment being evaluated.
        eval_mode (str): The mode of evaluation, such as 'test' or 'validation'.
        train (dict): Training-specific settings, including batch size and metrics configurations.
        device (str): The computational device to use (e.g., 'cuda' or 'cpu').
        gamelist (list): A list of games being evaluated.
    """
    logdir: str
    exp_name: str
    val: dict
    test: dict
    train: dict
    device: str
    gamelist: list

class SpaceEval:

    ap_results_none_dict = {
            'all': ({'adjusted_mutual_info_score': np.nan, 'adjusted_rand_score': np.nan}, "dummy_path",
                    {'few_shot_accuracy_with_1': np.nan, 'few_shot_accuracy_with_4': np.nan, 'few_shot_accuracy_with_16': np.nan,
                     'few_shot_accuracy_with_64': np.nan, 'few_shot_accuracy_cluster_nn': np.nan}),
            'moving': ({'adjusted_mutual_info_score': np.nan, 'adjusted_rand_score': np.nan}, "dummy_path",
                       {'few_shot_accuracy_with_1': np.nan, 'few_shot_accuracy_with_4': np.nan,
                        'few_shot_accuracy_with_16': np.nan,
                        'few_shot_accuracy_with_64': np.nan, 'few_shot_accuracy_cluster_nn': np.nan}),
            'relevant': ({'adjusted_mutual_info_score': np.nan, 'adjusted_rand_score': np.nan}, "dummy_path",
                         {'few_shot_accuracy_with_1': np.nan, 'few_shot_accuracy_with_4': np.nan,
                          'few_shot_accuracy_with_16': np.nan,
                          'few_shot_accuracy_with_64': np.nan, 'few_shot_accuracy_cluster_nn': np.nan})
        }

    def __init__(self, cfg: EvalConfig, tb_writer, eval_mode):
        self.eval_mode = eval_mode
        METRICS_LOGDIR = f'{cfg.logdir}/{cfg.exp_name}/{eval_mode}_metrics.csv'
        self.eval_file_path = METRICS_LOGDIR
        self.first_eval = True
        self.data_subset_modes = ['all','relevant']
        self.tb_writer = tb_writer
        self.file_writer = EvalWriter(cfg, tb_writer, self.data_subset_modes, self.eval_mode, self.eval_file_path)
        self.cfg = cfg

    def log_ap_metrics(self, APs: list, data_subset_mode: str, global_step: int):
        """
        Logs average precision (AP) metrics to TensorBoard.
    
        Args:
            APs (list): List of AP values.
            data_subset_mode (str): Subset of data being evaluated ('all', 'relevant').
            global_step (int): Current global step for logging.
        """
        for ap, thres in zip(APs[1::4], THRESHOLDS["AP_IOU"][1::4]):
            self.tb_writer.add_scalar(f'val_aps_{data_subset_mode}/ap_{thres:.1}', ap, global_step)
        self.tb_writer.add_scalar(f'{data_subset_mode}/ap_avg_0.5', APs[len(APs) // 2], global_step)
        self.tb_writer.add_scalar(f'{data_subset_mode}/ap_avg_up', np.mean(APs[len(APs) // 2:]), global_step)
        self.tb_writer.add_scalar(f'{data_subset_mode}/ap_avg', np.mean(APs), global_step)
    
    @torch.no_grad()
    # @profile
    def eval(self, model, global_step, cfg,):
        """
        Evaluation. This includes:
            - mse evaluated on dataset
            - ap and accuracy evaluated on dataset
            - cluster metrics evaluated on dataset
        :return:
        """
        results = {}

        if self.first_eval:
            self.first_eval = False
            if os.path.exists(self.eval_file_path):
                os.remove(self.eval_file_path)
            self.file_writer.write_header()

        # Only create latents if not using pre-computed ones
        if not cfg.eval.get('use_precomputed_latents', False):
            print('Creating latent dataset...')
            create_latent_dataset(cfg, self.eval_mode, model=model)
            print('Done creating latent dataset.')
        
        results = self.core_eval_code(global_step, cfg, model)
        return results

    def core_eval_code(self, global_step, cfg, model):
        results_collector = {}
        with open(self.eval_file_path, "a") as file:
            self.file_writer.write_metric(None, global_step, global_step, use_writer=False)
            if 'cluster' in cfg.get(self.eval_mode).metrics:
                results = self.eval_clustering(global_step, cfg, self.eval_mode)
                results_collector.update(results)
                if cfg.train.log:
                    pp = pprint.PrettyPrinter(depth=2)
                    for res in results:
                        print("Cluster Result:")
                        pp.pprint(results[res])
            if 'mse' in cfg.get(self.eval_mode).metrics:
                mse = self.eval_mse(model, global_step, cfg)
                results_collector.update({'mse': mse})
            if 'ap' in cfg.get(self.eval_mode).metrics:
                results = self.eval_ap_and_acc(global_step, self.eval_mode)
                results_collector.update(results)
                if cfg.train.log:
                    results = {k2: v2[len(v2) // 4] if isinstance(v2, list) or isinstance(v2, np.ndarray) else v2 for
                               k2, v2, in
                               results.items()}
                    pp = pprint.PrettyPrinter(depth=2)
                    print("AP Result:")
                    pp.pprint({k: v for k, v in results.items() if "iou" not in k})
            file.write("\n")
        return results_collector

    @torch.no_grad()
    def eval_ap_and_acc(self, global_step, eval_mode):
        """
        Evaluate ap and accuracy
        :return: result_dict
        """
        result_dict = ApAndAccEval(self.cfg).eval_ap_and_acc(self.data_subset_modes, eval_mode)
        for data_subset_mode in self.data_subset_modes:
            APs = result_dict[f'APs_{data_subset_mode}']
            # only logging
            self.log_ap_metrics(APs, data_subset_mode, global_step)
            # writing to file (and potentially logging)
            for ap in APs:
                self.file_writer.write_metric('ignored', ap, global_step, use_writer=False)
            metrics = ['accuracy', 'perfect', 'overcount', 'undercount', 'error_rate', 'precision', 'recall']
            self.file_writer.write_metrics(metrics, data_subset_mode, result_dict, global_step)
            # recall for different types of objects
            complete_label_list = label_list_for(self.cfg.gamelist[0])
            for label in range(len(complete_label_list)):
                value = result_dict[f'recall_label_{label}_{data_subset_mode}'] if f'recall_label_{label}_{data_subset_mode}' in result_dict else np.nan
                self.file_writer.write_metric(f'{data_subset_mode}/recall_label_{label}', value, global_step)
            for i, threshold in enumerate(THRESHOLDS["PREC_REC"]):
                self.file_writer.write_metric(f'{data_subset_mode}/precision_at_{threshold}', result_dict[f'precisions_{data_subset_mode}'][i], global_step)
                self.file_writer.write_metric(f'{data_subset_mode}/recall_at_{threshold}', result_dict[f'recalls_{data_subset_mode}'][i], global_step,
                                  make_sep=(data_subset_mode != 'relevant') or (i != len(THRESHOLDS["PREC_REC"]) - 1))
      
        return result_dict

    @torch.no_grad()
    def eval_mse(self, model, global_step, cfg):
        """
        Evaluate MSE by reconstructing images from latent space and comparing with original images.
        """
        if cfg.eval.get('use_precomputed_latents', False):
            print('Skipping MSE computation when using pre-computed latents')
            return np.nan

        num_batches = cfg.get(self.eval_mode).num_samples.mse // cfg.train.batch_size
        print(f'Computing MSE on subset of {cfg.train.batch_size * num_batches} samples...')
        
        # Load the dataset for evaluation
        dataset = Atari_Z_What(cfg, self.eval_mode, return_keys=["imgs"])
        dataloader = get_dataloader(cfg, self.eval_mode, dataset, no_shuffle_overwrite=True)

        metric_logger = MetricLogger()

        model.eval()  # Set model to evaluation mode

        for i, data_dict in enumerate(dataloader):
            if i >= num_batches:
                break

            # Get input images
            image_tensor = data_dict["imgs"].to(cfg.device)

            # Forward pass through the model to get reconstructed images
            with torch.no_grad():
                _, log = model(image_tensor)

             # Extract the reconstructed images from log
            recon_imgs = log['y']  # Reconstructed images

            # combine T and B dimensions such that image_tensor and recon_imgs have the same shape
            image_tensor = image_tensor.view(-1, *image_tensor.shape[2:])
            # Compute MSE loss between original images and reconstructed images
            mse_loss = F.mse_loss(recon_imgs, image_tensor)

            # Log the MSE
            metric_logger.update(mse=mse_loss.item())

        mse = metric_logger['mse'].global_avg
        self.file_writer.write_metric(f'all/mse', mse, global_step=global_step)
        
        print("MSE result: ", mse)
        return mse

    @torch.no_grad()
    def eval_clustering(self, global_step, cfg, eval_mode):
        """
        Evaluate clustering
        :return: result_dict
        """
        print('Computing clustering and few-shot linear classifiers...')
        results = ClusteringEval(cfg).eval_clustering(self.data_subset_modes, eval_mode)
        if (None, None, None) in results.values():
            results = self.ap_results_none_dict
        for name, (result_dict, img_path, few_shot_accuracy) in results.items():
            try:
                self.tb_writer.add_image(f'Clustering {name.title()}', np.array(Image.open(img_path)), global_step,
                                 dataformats='HWC')
            except Exception as e:
                print(f"Error adding clustering image {img_path}: {e}")

            metrics = [f"few_shot_accuracy_with_{train_objects_per_class}" for train_objects_per_class in [1, 4, 16, 64]] \
                + ['few_shot_accuracy_cluster_nn', 'adjusted_mutual_info_score', 'adjusted_rand_score']
            combined_dict = {**result_dict, **few_shot_accuracy}
            self.file_writer.write_metrics(metrics, name, combined_dict, global_step, class_name_part_of_key=False)
        return results

class EvalWriter:
    def __init__(self, cfg: EvalConfig, tb_writer: SummaryWriter, data_subset_modes, eval_mode, eval_file_path):
        self.cfg = cfg
        self.tb_writer = tb_writer
        self.data_subset_modes = data_subset_modes
        self.eval_mode = eval_mode
        self.eval_file_path = eval_file_path

    def write_metric(self, tb_label, value, global_step, use_writer=True, make_sep=True):
        if use_writer:
            self.tb_writer.add_scalar(tb_label, value, global_step)
        self.write_to_file(value, make_sep)
    
    def write_to_file(self, value, make_sep=True):
        with open(self.eval_file_path, "a") as file:
            file.write(f'{value}' + (";" if make_sep else ""))
    
    def write_metrics(self, metrics, class_name, metrics_dict, global_step, class_name_part_of_key=True):
        for metric in metrics:
            key_string = f'{metric}_{class_name}' if class_name_part_of_key else metric
            self.write_metric(f'{class_name}/{metric}', metrics_dict[key_string], global_step)

    def write_header(self):
        columns = self.generate_header_list()
        with open(self.eval_file_path, "w") as file:
            file.write(";".join(columns))
            file.write("\n")

    def generate_header_list(self):
        columns = ['global_step']
        if 'cluster' in self.cfg.get(self.eval_mode).metrics:
            columns.extend(self.get_cluster_header())
        if 'mse' in self.cfg.get(self.eval_mode).metrics:
            columns.append('mse')
        if 'ap' in self.cfg.get(self.eval_mode).metrics:
            columns.extend(self.get_ap_header())
        return columns

    def get_cluster_header(self):
        column_endings = [f"few_shot_accuracy_with_{train_objects_per_class}" for train_objects_per_class in [1, 4, 16, 64]] \
                + ['few_shot_accuracy_cluster_nn', 'adjusted_mutual_info_score', 'adjusted_rand_score']
        column_starts = self.data_subset_modes
        columns = [f"{class_name}_{column_ending}" for class_name in column_starts for column_ending in column_endings]
        return columns

    def get_ap_header(self):
        complete_label_list = label_list_for(self.cfg.gamelist[0])
        column_endings = [f"ap_{iou_t:.2f}" for iou_t in THRESHOLDS["AP_IOU"]] + \
            ['accuracy', 'perfect', 'overcount', 'undercount', 'error_rate', 'precision', 'recall'] + \
            [f"recall_label_{label}" for label in range(len(complete_label_list))] + \
            [f"precision_{thres:.2f}" for thres in THRESHOLDS["PREC_REC"]] + \
            [f"recall_{thres:.2f}" for thres in THRESHOLDS["PREC_REC"]]
        column_starts = self.data_subset_modes
        columns = [f"{class_name}_{column_ending}" for class_name in column_starts for column_ending in column_endings]
        return columns
