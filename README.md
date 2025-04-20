# Slot+MOC:
This repository contains the code for runinng MOC applied to SLOT. You can currently train and visualize bounding boxes for object detection in Pong.

**Sections**
- Options for running the code
  - Dockerfile (recommended)
  - Jupyter Notebook
- Trained model weights

<br>

**Clone repo and cd to root directory**
- <code>git clone </code> <br>
- <code>cd ox4rl_slot_attention</code>

<br>
<br>

**Options for running the code**

*Dockerfile*

We have two docker images one for training and another one for visualization.
You can either use train as the target or visualize, depending on whether you want to train the model from scratch or use our pretrained version to visualize bounding boxes.<br>
 The first commands are for building the images  <br>

```docker build -t ox4rl-slot-trainer --target trainer -f Dockerfile-slot .``` <br>
```docker build -t ox4rl-slot-visualizer --target visualizer -f Dockerfile-slot .``` <br>

Before running the containers the following folders need to be generated.

<code>mkdir -p visualizations/slot <br>
mkdir -p output/runs
</code>

And the second set of command are for running the containers. <br>

<code>
docker run --gpus all \
  -v $(pwd)/output/runs:/app/runs \
  --name slot_training \
  --rm \
  ox4rl-slot-trainer \
  --exp_name slot
</code> <br>

<code>
docker run --gpus all \
  -v $(pwd)/output/runs:/app/runs \
  -v $(pwd)/visualizations/slot:/app/visualizations/slot \
  --name slot_training_visualizer \
  --rm \
  ox4rl-slot-visualizer \
  --exp_name slot
</code>
<br>
<br>

*Jupyter Notebook*

You can find a Notebook named Slot_MOC_Pong which you can follow to create a conda env with the required dependencies and run the code to train and visualize the model. However, we recommend using the docker approach as the paths in the Jupyter Notebook are not up to date. 


**Trained model weights**

Find the already trained model weights at
```
ox4rl
├── epoch_1000_final.ckpt
├── ox4rl
```


Please consider that the model has only been trained on Pong.
