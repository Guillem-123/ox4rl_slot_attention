# Slot+MOC

This repository contains the code for running MOC applied to SLOT. You can currently train and visualize bounding boxes for object detection in Pong.

## Getting Started

### Clone Repository

```bash
git clone https://github.com/Guillem-123/ox4rl_slot_attention.git
cd ox4rl_slot_attention
```

## Running Options

### Docker (Recommended)

We have two Docker images: one for training and another for visualization.

#### 1. Build Docker Images

```bash
# For training
docker build -t ox4rl-slot-trainer --target trainer -f Dockerfile-slot .

# For visualization
docker build -t ox4rl-slot-visualizer --target visualizer -f Dockerfile-slot .
```

#### 2. Create Required Directories

```bash
mkdir -p visualizations/slot
mkdir -p output/runs
```

#### 3. Run Containers

For training:
```bash
docker run --gpus all \
  -v $(pwd)/output/runs:/app/runs \
  --name slot_training \
  --rm \
  ox4rl-slot-trainer \
  --exp_name slot
```

For visualization:
```bash
docker run --gpus all \
  -v $(pwd)/output/runs:/app/runs \
  -v $(pwd)/visualizations/slot:/app/visualizations/slot \
  --name slot_training_visualizer \
  --rm \
  ox4rl-slot-visualizer \
  --exp_name slot
```

### Jupyter Notebook

You can find a notebook named `Slot_MOC_Pong` which guides you through creating a conda environment with the required dependencies and running the code to train and visualize the model. 

> **Note:** We recommend using the Docker approach as the paths in the Jupyter Notebook are not up to date.

## Pretrained Model Weights

Trained model weights are available in the following structure:

```
ox4rl
├── epoch_1000_final.ckpt
├── ox4rl
```

Please note that the model has only been trained on Pong.
