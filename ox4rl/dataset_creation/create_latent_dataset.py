import os

from ox4rl.models.space.postprocess_latent_variables import retrieve_latent_repr_from_logs
import os.path as osp
import torch
from tqdm import tqdm
from ox4rl.dataset import get_dataloader
from ox4rl.dataset.atari_dataset import Atari_Z_What
from ox4rl.models.space.inference_space import WrappedSPACEforInference
from ox4rl.training.checkpointing.loading import load_model

def create_latent_dataset(cfg, dataset_mode = "test", model=None):
    game = cfg.gamelist[0]

    # Determine the source path for latents
    if cfg.eval.get('use_precomputed_latents', False):
        # Use the training validation latents path
        base_path = osp.join(cfg.latentsdir, cfg.exp_name, f'step_{cfg.eval.step:05d}')
    else:
        # Use the original evaluation path
        base_path = osp.join(cfg.dataset_roots.ATARI, game, "latents", dataset_mode)

    if cfg.eval.get('use_precomputed_latents', False):
        print(f'Using pre-computed latents from: {base_path}')
        return

    if model is None:
        model, _, _, _, _ = load_model(cfg, mode="eval")

    base_path = osp.join(cfg.dataset_roots.ATARI, game, "latents", dataset_mode)
    os.makedirs(base_path, exist_ok=True)

    create_latent_dataset_with_more_options(cfg, dataset_mode, model, base_path)

def create_latent_dataset_with_more_options(cfg, dataset_mode, model, base_path, number_of_data_points = None):
    game = cfg.gamelist[0]
    dataset = Atari_Z_What(cfg, dataset_mode, return_keys = ["imgs"])
    dataloader = get_dataloader(cfg, dataset_mode, dataset, no_shuffle_overwrite=True)

    base_path = osp.join(cfg.dataset_roots.ATARI, game, "latents", dataset_mode)
    os.makedirs(base_path, exist_ok=True)

    z_wheres = []
    z_pres_probs = []
    z_whats = []

    B = dataloader.batch_size
    T = dataset.T

    model = WrappedSPACEforInference(model)

    print("start inference and saving")
    for i, data_dict in enumerate(tqdm(dataloader)):
        if number_of_data_points is not None and i >= number_of_data_points:
            break

        image_tensor = data_dict["imgs"]
        image_tensor = image_tensor.to(cfg.device)
        with torch.no_grad():
            space_log = model(image_tensor)
        z_where, _, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)

        z_where = z_where.to(cfg.device)
        z_pres_prob = z_pres_prob.to(cfg.device)
        z_what = z_what.to(cfg.device)

        infix = f"0to{T-1}" if T > 1 else "0"
        for b in range(B):
            batch_index = i * B + b
            index = b * T
            torch.save(z_where[index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_where.pt"))    
            torch.save(z_pres_prob[index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_pres_prob.pt"))
            torch.save(z_what[index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_what.pt"))
    print("Finished inference and saving")

    # print("start inference")
    # for i, data_dict in enumerate(tqdm(dataloader)):
    #     if number_of_data_points is not None and i >= number_of_data_points:
    #         break

    #     image_tensor = data_dict["imgs"]
    #     image_tensor = image_tensor.to(cfg.device)
    #     with torch.no_grad():
    #         #_ , space_log = model(image_tensor, global_step=global_step)
    #         space_log = model(image_tensor)
    #     # (B*T, N, 4), (B*T, N,), (B*T, N), (B*T, N, 32)
    #     z_where, _, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)

    #     z_where = z_where.to(cfg.device)
    #     z_pres_prob = z_pres_prob.to(cfg.device)
    #     z_what = z_what.to(cfg.device)

    #     z_wheres.append(z_where)
    #     z_pres_probs.append(z_pres_prob)
    #     z_whats.append(z_what)
    # print("finished inference")

    # print("start saving")
    # infix = f"0to{T-1}" if T > 1 else "0"
    # for i in range(len(z_wheres)):
    #     for b in range(B):
    #         batch_index = i * B + b
    #         index = b * T
    #         torch.save(z_wheres[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_where.pt"))    
    #         torch.save(z_pres_probs[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_pres_prob.pt"))
    #         torch.save(z_whats[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_what.pt"))
    # print("Finished saving")

if __name__ == "__main__":
    import argparse
    from ox4rl.utils.load_config import get_config_v2
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--dataset-mode", required=True, help="dataset mode", type=str)
    config_path = parser.parse_args().config_file
    cfg = get_config_v2(config_path)
    dataset_mode = parser.parse_args().dataset_mode
    create_latent_dataset(cfg, dataset_mode)