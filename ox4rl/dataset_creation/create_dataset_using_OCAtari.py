from typing import Any, Optional

from enum import Enum

from pathlib import Path

from PIL import Image
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import random

from ox4rl.dataset_creation import bb
from ox4rl.dataset_creation.motion import mode
from ox4rl.dataset_creation.motion.motion_processing import BoundingBoxes, Identity, ZWhereZPres, set_color_hist, set_special_color_weight
from ox4rl.utils.niceprint import pprint as print

#OCAtari
import gymnasium as gym
from ocatari.core import OCAtari
import random
"""
If you look at the atari_env source code, essentially:

v0 vs v4: v0 has repeat_action_probability of 0.25 (meaning 25% of the time the
previous action will be used instead of the new action),
while v4 has 0 (always follow your issued action)
Deterministic: a fixed frameskip of 4, while for the env without Deterministic,
frameskip is sampled from (2,5)
There is also NoFrameskip-v4 with no frame skip and no action repeat
stochasticity.
"""

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def draw_action(self):
        return random.randint(0, self.env.nb_actions-1)

class BaseDetectionModel(Enum):
    SPACE="SPACE"
    SLOT_ATTENTION="SLOT"

class OCAtariDatasetCreator:

    REQ_CONSECUTIVE_IMAGE = 20 # defines how many steps are taken before the last T frames are saved
    T = 4 # number of consecutive frames, ATTENTION: most of the code is written for T=4, so it might not work for other values


    def __init__(
        self,
        directory: str,
        game_str: str = "SpaceInvaders",
        compute_root_images: bool = False,
        no_color_hist: bool = False,
        render: bool = False,
        stacks: bool = True,
        compute_bounding_boxes: bool = True,
        vis: bool = False,
        base_detection_model: BaseDetectionModel = BaseDetectionModel.SPACE
    ):
        self._directory = directory
        self._game_str = game_str
        self._should_compute_root_images = compute_root_images
        self._no_color_hist = no_color_hist
        self._render = render
        self._stacks = stacks
        self._compute_bounding_boxes = compute_bounding_boxes
        self._vis = vis
        self._base_detection_model = base_detection_model

        assert OCAtariDatasetCreator.T <= OCAtariDatasetCreator.REQ_CONSECUTIVE_IMAGE, "T must be smaller or equal to REQ_CONSECUTIVE_IMAGE"

        self._directory_sizes = {
            "train": 8192,
            "test": 1024,
            "validation": 1024
        }

        self._agent, _, _ = self._configure_OCAtari_environment()
        self._init_output_directories()

    def _init_output_directories(self):
        self._data_base_directory = Path('..','aiml_atari_data')
        self._mode_base_directory = Path('..','aiml_atari_data')

        self._create_dir_if_not_exists(self._data_base_directory)
        self._create_dir_if_not_exists(self._mode_base_directory)

        self.game_base_directory = Path(self._data_base_directory, f"{self._game_str}")

        self.rgb_directory = Path(self.game_base_directory, 'rgb', self._directory)
        self.bgr_directory = Path(self.game_base_directory, 'space_like', self._directory)
        self.bb_directory = Path(self.game_base_directory, 'bb', self._directory)
        self.mode_directory = Path(self.game_base_directory, 'mode', self._directory)

        self.vis_directory = None
        if self._vis:
            self.vis_directory = Path(self.game_base_directory, 'vis', self._directory)
        
        self._create_dir_if_not_exists(self.game_base_directory)
        self._create_dir_if_not_exists(self.rgb_directory)
        self._create_dir_if_not_exists(self.bgr_directory)
        self._create_dir_if_not_exists(self.bb_directory)
        self._create_dir_if_not_exists(self.mode_directory)

        if self.vis_directory:
            self._create_dir_if_not_exists(Path(self.vis_directory, 'BoundingBox'))
            self._create_dir_if_not_exists(Path(self.vis_directory, 'Mode'))
    
    def run(self):
        visualization_modes = []
        visualizations_bb = []

        if self._vis:
            visualization_modes.append(Identity(vis_path=self.vis_directory, motion_type="Mode", max_vis=50, every_n=1))
            visualization_modes.append(ZWhereZPres(vis_path=self.vis_directory, motion_type="Mode", max_vis=20, every_n=2))
            visualizations_bb.append(BoundingBoxes(vis_path=self.vis_directory, motion_type='BoundingBox', max_vis=20, every_n=1))

        # take some steps to not start at the beginning of the game (might be unnecessary)
        self._warm_up_agent(n_actions=100)

        # compute the root image i.e. mode as the background
        if self._should_compute_root_images:
            self._compute_root_images()
            return None
    
        self._compute_images(
            visualization_modes=visualization_modes,
            visualization_bb=visualizations_bb
        )
        
    def _compute_images(
        self,
        visualization_modes: list,
        visualization_bb: list
    ):
        self._take_some_agent_steps(
            n_steps=50,
            reset_env=False
        )

        root_mode = self._load_root_mode()

        if root_mode is None:
            return None

        self._set_color_hist(root_mode=root_mode)
        
        self._create_dataset(
            n_images=self._directory_sizes[self._directory],
            root_mode=root_mode,
            visualization_modes=visualization_modes,
            visualization_bb=visualization_bb
        )

    def _create_dataset(
        self,
        n_images: int,
        root_mode,
        visualization_modes: list,
        visualization_bb: list
    ):
        progress_bar = tqdm(total=n_images)

        image_count = 0
        consecutive_images = []
        consecutive_images_info = []

        while True: 
            obs, reward, done, truncated, info = self._take_agent_action()

            game_objects = [
                (go.y, go.x, go.h, go.w, "S" if go.hud else "M", go.category)
                for go in sorted(self._agent.env.objects, key=lambda o: str(o))
            ]

            info["bbs"] = game_objects

            if (obs == 0).all():
                continue

            if self._render:
                self._agent.env.render()

            consecutive_images += [obs]
            consecutive_images_info.append(info)

            if len(consecutive_images) == OCAtariDatasetCreator.REQ_CONSECUTIVE_IMAGE:
                if self._base_detection_model == BaseDetectionModel.SPACE:
                    self._save_consecutive_images_for_space(
                        consecutive_images=consecutive_images,
                        consecutive_images_info=consecutive_images_info,
                        image_count=image_count,
                        root_mode=root_mode,
                        visualization_modes=visualization_modes,
                        visualization_bb=visualization_bb
                    )
                elif self._base_detection_model == BaseDetectionModel.SLOT_ATTENTION:
                    self._save_consecutive_images_for_slot_attention()
                
                while done or truncated:
                    obs, reward, done, truncated, info = self._take_some_agent_steps(
                        n_steps=10,
                        reset_env=True
                    )

                consecutive_images, consecutive_images_info = [], []
                
                progress_bar.update(1)

                image_count += 1
            else:
                while done or truncated:
                    obs, reward, done, truncated, info = self._take_some_agent_steps(
                        n_steps=10,
                        reset_env=True
                    )

                    consecutive_images, consecutive_images_info = [], []
        
            if image_count == n_images:
                break
    
    def _save_consecutive_images_for_space(
        self,
        consecutive_images: list,
        consecutive_images_info: list,
        image_count: int,
        root_mode,
        visualization_modes: list,
        visualization_bb: list,
    ):
        space_stack = []
        
        for frame in consecutive_images[:-OCAtariDatasetCreator.T]:
            space_stack.append(frame)
        
        resize_stack = []
        
        for i, (frame, img_info) in enumerate(zip(consecutive_images[-OCAtariDatasetCreator.T:], consecutive_images_info[-OCAtariDatasetCreator.T:])):
            space_stack.append(frame)
            frame_space = Image.fromarray(frame[:, :, ::-1], 'RGB').resize((128, 128), Image.LANCZOS)
            resize_stack.append(np.array(frame_space))
            frame_space.save(f'{self.bgr_directory}/{image_count:05}_{i}.png')
            img = Image.fromarray(frame, 'RGB')
            img.save(f'{self.rgb_directory}/{image_count:05}_{i}.png')
            bb.save(frame_space, img_info, f'{self.bb_directory}/{image_count:05}_{i}.csv', visualization_bb)
        
        resize_stack = np.stack(resize_stack)
        space_stack = np.stack(space_stack)
        
        mode.save(
            space_stack,
            f'{self.mode_directory}/{image_count:05}_{{}}.pt',
            visualization_modes,
            mode=root_mode,
            space_frame=resize_stack
        )
    
    def _save_consecutive_images_for_slot_attention(self):
        raise NotImplementedError("")

    def _load_root_mode(self):
        mode_path = Path(self._mode_base_directory, f"{self._game_str}", "background", "mode.png")

        if not mode_path.exists():
            return None

        return np.asanyarray(
            Image.open(
                mode_path
            )
        )[:, :, :3]

    def _set_color_hist(self, root_mode):
        set_color_hist(root_mode)

        if "Pong" in self._game_str:
            set_special_color_weight(15406316, 8)
        
        if "Airraid" in self._game_str:
            set_special_color_weight(0, 20000)
        
        if "Riverraid" in self._game_str:
            set_special_color_weight(3497752, 20000) 
        
    def _warm_up_agent(self, n_actions: int):
        for _ in range(n_actions):
            obs, reward, done, truncated, info = self._take_agent_action()

            if done or truncated:
                self._agent.env.reset()
            
    def _compute_root_images(self, limit:int =100):
        imgs = []
        
        while len(imgs) < limit:
            obs, reward, done, truncated, info = self._take_agent_action()

            progress_bar = tqdm(total=limit)

            if np.random.rand() < 0.25:
                imgs.append(obs)
                progress_bar.update(1)
            
            if done or truncated:
                self._agent.env.reset()
                for _ in range(100):
                    obs, reward, done, truncated, info = self._take_agent_action()
        
        self._save_root_images(imgs=imgs)
    
    def _save_root_images(self, imgs):
        img_arr = np.stack(imgs)
        mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=img_arr).astype(np.uint8)

        save_dir = Path(self._data_base_directory, f"{self._game_str}", "background")
        self._create_dir_if_not_exists(save_dir)
        
        frame = Image.fromarray(mode)
        frame.save(Path(save_dir, "mode.png"))

        print("blue", f"Saved mode.png in {save_dir}")
    
    def _take_agent_action(self):
        action = self._agent.draw_action()
        obs, reward, done, truncated, info = self._agent.env.step(action)
        return obs, reward, done, truncated, info
    
    def _take_some_agent_steps(self, n_steps: int, reset_env: bool = False):
        if reset_env:
            self._agent.env.reset()

        for _ in range(n_steps):
             _, _, _, _, _ = self._take_agent_action()
        
        return self._take_agent_action()

    def _configure_OCAtari_environment(self):
        env = OCAtari(
            self._game_str,
            mode="ram", hud=False, obs_mode="ori"
        )

        self._game_str = env.spec.id

        env.reset()
        observation, info = env.reset()
        
        self._make_deterministic(
            0 if self._directory == "train" else 1 if self._directory == "validation" else 2,
            env
        )

        agent = RandomAgent(env=env)

        return agent, observation, info
    
    def _make_deterministic(self, seed, mdp, states_dict=None):
        if states_dict is None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"Set all environment deterministic to seed {seed}")
        else:
            np.random.set_state(states_dict["numpy"])
            torch.random.set_rng_state(states_dict["torch"])
            mdp.env.env.np_random.set_state(states_dict["env"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"Reset environment to recovered random state ")

    def _create_dir_if_not_exists(self, directory: Path):
        if not directory.exists():
            directory.mkdir(parents=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the dataset for a specific game of ATARI Gym')
    parser.add_argument('-g', '--game', type=str, help='An atari game', default='SpaceInvaders')
    parser.add_argument('--compute_root_images', default=False, action="store_true", help='instead compute the mode of the images')
    parser.add_argument('--no_color_hist', default=False, action="store_true", help='use the color_hist to filter')
    parser.add_argument('--render', default=False, action="store_true", help='renders the environment')
    parser.add_argument('-s', '--stacks', default=True, action="store_false", help='should render in correlated stacks of T (defined in code) frames')
    parser.add_argument('--bb', default=True, action="store_false", help='should compute bounding_boxes')
    parser.add_argument('-f', '--folder', type=str, choices=["train", "test", "validation"], required=True, help='folder to write to: train, test or validation')
    parser.add_argument('--vis', default=False, action="store_true", help='creates folder vis with visualizations which can be used for debugging')
    parser.add_argument('-m', '--model', type=str, choices=["SPACE", "SLOT"], required=True, help='Name of base detection model.')
    args = parser.parse_args()

    ocatari_dataset_creator = OCAtariDatasetCreator(
        directory=args.folder,
        game_str=args.game,
        compute_root_images=args.compute_root_images,
        no_color_hist=args.no_color_hist,
        render=args.render,
        stacks=args.stacks,
        compute_bounding_boxes=args.bb,
        vis=args.vis,
        base_detection_model=BaseDetectionModel[args.model]
    )

    ocatari_dataset_creator.run()
