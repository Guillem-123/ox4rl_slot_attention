from pathlib import Path
from ox4rl.eval.eval_model.space_eval import SpaceEval
from ox4rl.dataset import get_dataloader
from ox4rl.training.checkpointing.loading import print_info, load_model
from ox4rl.training.metric_logger import MetricLogger
import os
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from torch.nn.utils import clip_grad_norm_
from rtpt import RTPT
from tqdm import tqdm
from ox4rl.losses.moc_loss_4_space import MOCLoss
from ox4rl.dataset.atari_dataset import Atari_Z_What
from ox4rl.training.logging import Logger
from ox4rl.training.latents_logger import LatentsLogger


def train(cfg, rtpt_active=True):
    assert len(cfg.gamelist) == 1, "Only one game is supported for training."
    print_info(cfg)

    if rtpt_active:  # RTPT: Remaining Time to Process (library by AIML Lab used to rename your processes giving
        # information on who is launching the process, and the remaining time for it)
        rtpt = RTPT(name_initials='TRo', experiment_name='SPACE-Time', max_iterations=cfg.train.max_epochs)
        rtpt.start()

    # data loading
    dataset = Atari_Z_What(cfg, 'train', return_keys=["imgs", "motion", "motion_z_pres", "motion_z_where"])
    train_data_loader = get_dataloader(cfg, 'train', dataset)

    latents_logger = LatentsLogger(Path(cfg.latentsdir, cfg.exp_name))

    # model loading
    model, optimizer_fg, optimizer_bg, checkpointer, checkpoint = load_model(cfg, mode="train")
    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step'] + 1
    else:
        start_epoch = 0
        global_step = 0
    moc_loss_instance = MOCLoss()

    # prepare logging
    tb_writer, metric_logger, logger_tb_and_vis = load_loggers(cfg, global_step)

    # prepare evaluation
    if cfg.train.eval_on:
        evaluator = SpaceEval(cfg, tb_writer, eval_mode="validation")

    print(f"training on {len(dataset)} samples of {dataset.T} consecutive frames")
    print(f'Start training, Global Step: {global_step}, Start Epoch: {start_epoch} Max: {cfg.train.max_steps}')

    # initialize variables for training loop
    model.train()
    end_flag = False
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if end_flag:
            break

        data_time_start = time.perf_counter()
        for data_dict in tqdm(train_data_loader, desc=f"Epoch {epoch}"):
            data_time_end = time.perf_counter()
            data_time = data_time_end - data_time_start


            # eval on validation set
            if cfg.train.eval_on and (global_step % cfg.train.eval_every == 0):
                print('Validating...')

                val_time_start = time.perf_counter()
                eval_checkpoint = [model, optimizer_fg, optimizer_bg, epoch, global_step]
                results = evaluator.eval(model, global_step, cfg)
                
                checkpointer.save_best(
                    "precision_relevant",
                    results["precision_relevant"],
                    eval_checkpoint,
                    min_is_better=False
                )

                val_time_end = time.perf_counter()

                print('Validation takes {:.4f}s.'.format(val_time_end - val_time_start))

            if cfg.train.log_latents and (global_step % cfg.train.log_latents_every == 0): #TODO: add respective fields in configs
                _log_latents(
                    latents_logger=latents_logger,
                    cfg=cfg,
                    model=model,
                    global_step=global_step
                )

            print("Training...")
            # main training
            batch_time_start = time.perf_counter()
            model.train()
            img_stacks, motion, motion_z_pres, motion_z_where = data_dict["imgs"], data_dict["motion"], data_dict["motion_z_pres"], data_dict["motion_z_where"]
            img_stacks, motion, motion_z_pres, motion_z_where = img_stacks.to(cfg.device), motion.to(cfg.device), motion_z_pres.to(cfg.device), motion_z_where.to(cfg.device)
            base_loss, log = model(img_stacks, global_step)
            moc_loss, log = moc_loss_instance.compute_loss(motion, motion_z_pres, motion_z_where, log, global_step)
            loss = base_loss + moc_loss
            loss = loss.mean()  # In case of using DataParallel
            optimizer_fg.zero_grad(set_to_none=True)
            optimizer_bg.zero_grad(set_to_none=True)
            batch_time_end = time.perf_counter()
            batch_time = batch_time_end - batch_time_start
            metric_logger.update(data_time=data_time)
            metric_logger.update(batch_time=batch_time)
            metric_logger.update(loss=loss.item())
            loss.backward()
            if cfg.train.clip_norm:
                clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
            optimizer_fg.step()
            optimizer_bg.step()

            # logging
            if global_step % 20 == 0:  # print in console
                log_state(cfg, epoch, global_step, log, metric_logger)
            if global_step % cfg.train.print_every == 0:  # log in tensorboard
                log.update({
                    'loss': metric_logger['loss'].median,
                })
                logger_tb_and_vis.log(model, tb_writer, log, global_step, 'train', cfg, dataset)

            # checkpointing
            if global_step % cfg.train.save_every == 0:  # save checkpoint
                _save_checkpoint(
                    checkpointer=checkpointer,
                    model=model,
                    optimizer_fg=optimizer_fg,
                    optimizer_bg=optimizer_bg,
                    epoch=epoch,
                    global_step=global_step
                )

            global_step += 1

            # check ending condition
            if global_step > cfg.train.max_steps:
                end_flag = True

                # final evaluation on validation set
                if cfg.train.eval_on:
                    print('Final evaluation on validation set...')
                    final_eval_time_start = time.perf_counter()
                    evaluator.eval(model, global_step, cfg)
                    print('Validation takes {:.4f}s.'.format(time.perf_counter() - final_eval_time_start))

                break

            data_time_start = time.perf_counter()

        if rtpt_active:
            rtpt.step()


def load_loggers(cfg, global_step):
    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    
    tb_writer = SummaryWriter(
        log_dir=log_path,
        flush_secs=30,
        purge_step=global_step
    )  # tb refers to tensorboard

    metric_logger = MetricLogger()
    logger_tb_and_vis = Logger()

    return tb_writer, metric_logger, logger_tb_and_vis


def log_state(cfg, epoch, global_step, log, metric_logger):
    print()
    print(
        'exp: {}, epoch: {}, global_step: {}, '
        'loss: {:.2f}, z_what_loss_objects: {:.3f}, motion_loss_z_pres: {:.3f}, motion_loss_z_where: {:.3f}, motion_loss_alpha: {:.3f},'
        'batch time: {:.4f}s, data time: {:.4f}s'.format(
            cfg.exp_name, epoch + 1,global_step,
            metric_logger['loss'].median,
            torch.sum(log['z_what_loss_objects']).item(),
            torch.sum(log['flow_loss_z_pres']).item(), torch.sum(log['flow_loss_z_where']).item(), torch.sum(log['flow_loss_alpha_map']).item(),
            metric_logger['batch_time'].avg, metric_logger['data_time'].avg
        )
    )
    print()

def _save_checkpoint(
    checkpointer,
    model,
    optimizer_fg,
    optimizer_bg,
    epoch,
    global_step
):
    checkpointing_time_start = time.perf_counter()
    checkpointer.save_last(model, optimizer_fg, optimizer_bg, epoch, global_step)
    print('Saving checkpoint takes {:.4f}s.'.format(time.perf_counter() - checkpointing_time_start))

def _log_latents(latents_logger: LatentsLogger, cfg, model, global_step):
    latents_logger_time_start = time.perf_counter()

    latents_logger.save_latents(cfg, model, global_step, number_of_data_points = 8)

    print('Saving latents takes {:.4f}s.'.format(time.perf_counter() - latents_logger_time_start))


if __name__ == "__main__":
    import argparse
    from ox4rl.utils.load_config import get_config_v2
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    config_path = parser.parse_args().config_file
    # directory_path = './configs'  # Adjust this to your directory

    # current_directory = os.getcwd()
    # print(f"Current working directory: {current_directory}")
    # directory_path = "./configs/my_atari_pong.yaml"
    # List all files and directories
    # try:
    #     contents = os.listdir(directory_path)
    #     print(f"Contents of '{directory_path}':")
    #     for item in contents:
    #         print(item)
    # except FileNotFoundError:
    #     print(f"The directory '{directory_path}' does not exist.")

    cfg = get_config_v2(config_path)
    train(cfg)
