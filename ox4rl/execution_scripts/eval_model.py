from ox4rl.eval.eval_model.space_eval import SpaceEval
import os
from torch.utils.tensorboard import SummaryWriter
from ox4rl.training.checkpointing.loading import print_info, load_model

def eval(cfg):
    assert cfg.eval.checkpoint in ['best', 'last']
    assert cfg.eval.metric in ['ap_dot5', 'ap_avg']

    print_info(cfg)

    if cfg.eval.eval_ckpt:
        print(f"Loading specific checkpoint: {cfg.eval.eval_ckpt}")
    else:
        print(f"Loading {cfg.eval.checkpoint} checkpoint.")
        
    model, _, _, _, _ = load_model(cfg, mode="eval")

    print('Loading evaluator...')
    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    global_step = 100000
    tb_writer = SummaryWriter(log_dir=log_path, flush_secs=30,
                           purge_step=global_step) # tb refers to tensorboard
    evaluator = SpaceEval(cfg, tb_writer, eval_mode="test")

    model.eval()
    evaluator.eval(model, global_step, cfg)

    # # data loading
    # vis_logger = SpaceVis()
    # print(f"Dataset: {cfg.dataset}")
    # #print(f"Show mode: {cfg.show_mode}")
    # dataset = get_dataset(cfg, "train")
    #
    # os.makedirs(cfg.demodir, exist_ok=True)
    # img_path = osp.join(cfg.demodir, '{}.png'.format(cfg.exp_name))
    # vis_logger.show_vis(model, dataset, [0, 1, 2],  img_path, device=cfg.device)
    # print('The result image has been saved to {}'.format(osp.abspath(img_path)))




if __name__ == "__main__":
    import argparse
    from ox4rl.utils.load_config import get_config_v2
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    config_path = parser.parse_args().config_file
    cfg = get_config_v2(config_path)
    eval(cfg)