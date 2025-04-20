from pathlib import Path
import os
import torch

from ox4rl.dataset_creation.create_latent_dataset import create_latent_dataset_with_more_options
class LatentsLogger:
    def __init__(self, directory: Path):
        self._directory = directory

        self._create_dir_if_not_exists(
            path=self._directory
        )
    
    def _create_dir_if_not_exists(self, path: Path):
        if not path.exists():
            path.mkdir(parents=True)
    
    def save_latents(self, cfg, model, global_step, number_of_data_points):
        base_path = Path(self._directory, "step_" + str(global_step).zfill(5))
        try:
            os.makedirs(base_path, exist_ok=False)
        except FileExistsError:
            print(f"WARN: {base_path} already exists. Overwriting.")

        create_latent_dataset_with_more_options(cfg, "validation", model, base_path, number_of_data_points)