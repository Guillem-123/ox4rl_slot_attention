from ox4rl.execution_scripts.eval_model_and_classifier import eval_model_and_classifier
from ox4rl.execution_scripts.eval_classifier import eval_classifier
from ox4rl.execution_scripts.train_classifier import train_classifier
from ox4rl.utils.load_config import get_config_v2
import numpy as np
from ox4rl.execution_scripts.eval_model import eval
import os

models_path = "../tmp_models"
cfg_path_for_game = {
    #"pong": "configs/my_atari_pong_gpu.yaml",
	#"skiing": "configs/my_atari_skiing_gpu.yaml",
    #"seaquest": "configs/my_atari_seaquest_gpu.yaml",
    #"asterix": "configs/my_atari_asterix_gpu.yaml",
    #"boxing": "configs/my_atari_boxing_gpu.yaml",
    #"freeway": "configs/my_atari_freeway_gpu.yaml",
    "kangaroo": "configs/my_atari_kangaroo_gpu.yaml", #ERROR
    # "bowling": "configs/my_atari_bowling_gpu.yaml", #ERROR
}
for game in cfg_path_for_game.keys():
    cfg = get_config_v2(cfg_path_for_game[game])
    cfg_override_list = [
        # "exp_name", f"{game}_seed{seed}",
        # "resume_ckpt", f"{models_path}/{game}_seed{seed}/model_000005001.pth",
        # "logdir", f"{models_path}",
        "eval.eval_ckpt", f"../output/checkpoints/{game}/model_000005001.pth" if os.path.exists(f"../output/checkpoints/{game}/model_000005001.pth") else f"../output/checkpoints/{game}/model_000040001.pth",

    ]
    
    cfg.merge_from_list(cfg_override_list)
    import ipdb; ipdb.set_trace()
    eval(cfg)
    train_classifier(cfg)
    eval_classifier(cfg)
    eval_model_and_classifier(cfg)
    print("one iteration done")





