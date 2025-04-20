# SPACE+MOC:
This repository contains the code for MOC applied to SPACE (https://arxiv.org/abs/2001.02407). Here you can train discovery models to detect object in different Atari 2600 environments.

**Sections**
- Installation
- Trained model weights
- Dataset Creation
- Config Files
- Object Detection and Representation Model:
	- Loading the model
	- Training
	- Evaluation
- Object Classification
	- Training
	- Evaluation
- Object Tracking

**Installation**
- use python 3.8.12 or similar
- install requirements.txt
- **install the package locally with "pip install -e ." in the root directory of this repo**
- for installation with cuda usage on remote cluster of tu darmstadt, torch related packages might have to be installed separately (e.g., "pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113  -f  https://download.pytorch.org/whl/cu113/torch_stable.html")
- if you get a protobuf issue when running eval.py, the solution is to downgrade to 3.20

**Trained model weights**

You can optionally download already trained models for SPACE+MOC, which can be used for using this repo in conjunction with the SCoBots repo (https://github.com/k4ntz/SCoBots/ space_detectors branch).
You first need to get access to the data (https://hessenbox.tu-darmstadt.de/getlink/fi3BrmgYx9JyN54FokGkXwEQ/). Then you can download the SPACE+MOC model weights and the classifier for Pong, Boxing and Skiing
(scobots_spaceandmoc_detectors.tar.gz). 
Downloaded SPACE model weights and classifier should be placed in a directory called scobots_spaceandmoc_detectors in the root directory of this repo. This directory should be structured as follows (the structure is most likely already given when the scobots_spaceandmoc_detectors data is extracted):
```
scobots_spaceandmoc_detectors
├── boxing
│   ├── classifier
│   │   └── z-what-classifier_relevant_nn.joblib.pkl
│   └── space_weights
│       └── model_000005001.pth
├── pong
└── skiing
```

**Detector API Usage**
Follow the installation steps and download the trained model weights. Put the trained model inside ox4rl/ox4rl/ folder.
To see the usage of the detector API, check the file ox4rl/ox4rl/detector_api/test_api.py. Currently, the configurations are set to use of cpu.
NOT TESTED: To use gpu, modify the configuration files in ox4rl/ox4rl/configs/detector_api_configs/ to use gpu. (Take inspiration from the configs in ox4rl/ox4rl/configs).
Note: you can skip the rest of the README if you are only interested in using the detector API.


**Dataset Creation**

First root_images have to be created.
- `python -m ox4rl.dataset_creation.create_dataset_using_OCAtari -f train -m SPACE -g Pong --compute_root_images` # -f parameter is irrelevant when --compute_root_images is set

Then a dataset for each dataset_mode can be created.
- `python -m ox4rl.dataset_creation.create_dataset_using_OCAtari -f train -m SPACE -g Pong`
- `python -m ox4rl.dataset_creation.create_dataset_using_OCAtari -f validation -m SPACE -g Pong`
- `python -m ox4rl.dataset_creation.create_dataset_using_OCAtari -f test -m SPACE -g Pong`

With the parameter `m` the dataset creation for a specific model can be selected. At the moment are two models available:

- `SPACE` for the SPACE+MOC model
- `SLOT` for the Slot-Attention model

Generally, mode should be used instead of median (but this is already set as default, so this nothing to worry about).
The parameter --vis optionally creates also visualizations that can help to understand whether the data generation was successful.
The folders median, flow, rgb and vis are not required tor the training or evaluation. The sizes of train, validation and test set are specified in the python file but can easily be modified.

If consecutive images should be created (e.g., for evaluation of tracking), follow the same steps but use the create_consecutive_dataset.py instead.

If a trained model exists, a dataset of the latent variables that this model produces can be created:

`python3 main.py --task create_latent_dataset --config configs/config_file_name.yaml` TODO update this

Currently, this creates the latent dataset only for the "test" dataset_mode.

**Config Files**

Files used for config:
- args (passed via execution command)
- atari_\<gamename\>.yaml in src/configs 
- src/model/space/arch.py
- src/config.py: includes arch.py but specifies additional values

Handled via executing:
- get_config() in src/engine/utils.py:
  - prioritisation of values: args  > config > atari_<gamename>.yaml > config.py

**Object Detection and Representation Model**

Configs: most important parameters
```

model: 'tcspace'
resume: true
resume_ckpt: 'path/to/model_weights' # use '' to use last checkpoint as stored in model_list.pkl in checkpointdir

train:
  max_epochs: 1000
  max_steps: 10000
  # whichever is reached first

arch:
  motion_input: false
  motion: true
  motion_kind: 'mode' # choose between 'mode', 'flow' and 'median'


eval: # For engine/eval.py
  checkpoint: 'last'
  metric: ap_avg

gamelist: [
    'Pong-v0',
    ]
```

**Loading the model**

Like all methods from files within src, the code should be executed while being in the src directory and via the main file.

The model is loaded with `Checkpointer` in `src/utils/checkpointer.py`, while the config in `src/configs` (e.g. `my_atari_pong_gpu.yaml`) controls which model is loaded. Note that get_config() must called before somewhere!!!:

```python
# model loading
model = get_model(cfg)
model = model.to(cfg.device)
optimizer_fg, optimizer_bg = get_optimizers(cfg, model)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,)
start_epoch = 0
global_step = 0
checkpoint = checkpointer.load_last(cfg.resume, cfg.resume_ckpt, model, optimizer_fg, optimizer_bg, cfg.device)
if checkpoint:
	start_epoch = checkpoint['epoch']
	global_step = checkpoint['global_step'] + 1
if cfg.parallel:
	model = nn.DataParallel(model, device_ids=cfg.device_ids)
```
Note that some steps (e.g. passing the optimizers) are only necessary for training.

**Training the model**

Training a single model can be done with `train_model.py`:

`python -m ox4rl.execution_scripts.train_model --config-file .\configs\my_atari_pong.yaml`

Evaluation is run alongside training if the config `train.eval_on` is set to True.

**Evaluating the model**

`python -m ox4rl.execution_scripts.eval_model --config-file .\configs\my_atari_pong.yaml`

The file `src/configs/eval_cfg.py` specifies which metrics are used in the evaluation.

**Object Classification**

**Training the classifier**

The classifier can be created using
`python -m ox4rl.execution_scripts.train_classifier --config-file .\configs\my_atari_pong.yaml`.
This saves a classifier and a csv-file which specifies the mapping from the enumerated class labels to the index postion OC_ATARI lists.
These are saved in the folder where the space model weights for creating the latent dataset are stored.

**Evaluating a classifier**

The classifier can be evaluated using
`python -m ox4rl.execution_scripts.eval_classifier --config-file .\configs\my_atari_pong.yaml`.

TODO add relevant papers (SPACE, MOC, "DINSA" PAPER)


# REMARKS COPIED FROM ORIGINAL SPACE REPO

> [SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition](https://arxiv.org/abs/2001.02407)  

![spaceinv_with_bbox](figures/spaceinvaders.png)

[link to the original repo](https://github.com/zhixuan-lin/SPACE)

## General

Project directories:

* `src`: source code
* `output`: anything the program outputs will be saved here. These include
  * `output/checkpoints`: training checkpoints. Also, model weights with the best performance will be saved here
  * `output/logs`: tensorboard event files
  * `output/eval`: quantitative evaluation results
* `scripts`: some useful scripts for downloading things and showing demos

This project uses [YACS](https://github.com/rbgirshick/yacs) for managing experiment configurations. Configurations are specified with YAML files. These files are in `src/configs`.

## Dependencies

:bangbang: I made it work with PyTorch 1.8.0 (last version)

If you can use the default CUDA (>=10.2) version, then just use
```
pip3 install -U pip
pip3 install -r requirements.txt
```

## General Usage

**First, `cd src`.  Make sure you are in the `src` directory for all commands in this section. All paths referred to are also relative to `src`**.

The general command to run the program is (assuming you are in the `src` directory)

```
python main.py --task [TASK] --config [PATH TO CONFIG FILE] [OTHER OPTIONS TO OVERWRITE DEFAULT YACS CONFIG...]
```

Example usage of "[OTHER OPTIONS TO OVERWRITE DEFAULT YACS CONFIG...]":
These start training with GPU 0 (`cuda:0`). There some useful options that you can specify. For example, if you want to use GPU 5, 6, 7, and 8 and resume from checkpoint `../output/checkpoints/3d_room_large/model_000008001.pth`, you can run the following:

```
python main.py --task train --config configs/3d_room_large.yaml \
	resume True resume_ckpt '../output/checkpoints/3d_room_large/model_000008001.pth' \
	parallel True device 'cuda:5' device_ids '[5, 6, 7, 8]'
```
Other available options are specified in `config.py`.

**Training visualization**. Run the following

```
# Run this from the 'src' directory
tensorboard --bind_all --logdir '../output/logs' --port 8848
```

And visit `http://[your server's address]:8848` in your local browser.

## Issues

* For some reason we were using BGR images for our Atari dataset and our pretrained models can only handle that. Please convert the images to BGR if you are to test your own Atari images with the provided pretrained models.
* There is a chance that SPACE doesn't learn proper background segmentation for the 3D Room Large datasets. Due to the known [PyTorch reproducibity issue](https://pytorch.org/docs/stable/notes/randomness.html), we cannot guarantee each training run will produce exactly the same result even with the same seed. For the 3D Room Large datasets, if the model doesn't seem to be segmenting the background in 10k-15k steps, you may considering changing the seed and rerun (or not even changing the seed, it will be different anyway). Typically after trying 1 or 2 runs you will get a working version.

## Use SPACE for other tasks

If you want to apply SPACE to your own task (e.g., for RL), please be careful. Applying SPACE to RL is also our original intent, but we found that the model can sometimes be unstable and sensitive to hyperparameters and training tricks. There are several reasons:

1. **The definition of objects and background is ambiguous in many cases**. Atari is one case where objects are often well-defined. But in many other cases, it is not. For more complicated datasets, making SPACE separate foreground and background properly can be something non-trivial.
2. **Learning is difficult when object sizes vary a lot**. In SPACE, we need to set a proper prior for object sizes manually and that turn out to be crucial hyperparameter. For example, for the 10 Atari games we tested, objects are small and roughly of the same size. When object sizes vary a lot SPACE may fail.

That said, we are pleased to offer discussions and pointers if you need help (especially when fine-tuning it on your own dataset). We also hope this will facilitate future works that overcome these limitations.




