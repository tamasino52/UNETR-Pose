# UNETR-Pose
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ftamasino52%2FUNeTR-Pose&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=true)](https://hits.seeyoufarm.com)

3D Multi-person Pose Estimation in Multi-view Environment using 3D U-Net Transformer Networks.

I am not the author of below paper, but I made a multi-person multi-view pose estimator by applying the model of this paper. It's still being updated, so please debug it if necessary.

This model based on :
> [**UNETR: Transformers for 3D Medical Image Segmentation**],            
> Ali Hatamizadeh, Dong Yang, Holger Roth, Daguang Xu. 2021.
> *(https://arxiv.org/abs/2103.10504?context=cs.CV)*

<img src="/sample/validation_00000200_3d.png"><br>
<img src="/sample/validation_00000200_view_1_gt.jpg"><br>
<img src="/sample/validation_00000200_view_2_gt.jpg"><br>
<img src="/sample/validation_00000200_view_3_gt.jpg"><br>
<img src="/sample/validation_00000200_view_4_gt.jpg"><br>
<img src="/sample/validation_00000200_view_5_gt.jpg">

## Reference
This code based on https://github.com/microsoft/voxelpose-pytorch.

## Installation
1. Clone this repo, and we'll call the directory that you cloned multiview-multiperson-pose as ${POSE_ROOT}.
2. Install dependencies.

## Data preparation

### Shelf/Campus datasets
1. Download the datasets from http://campar.in.tum.de/Chair/MultiHumanPose and extract them under `${POSE_ROOT}/data/Shelf` and `${POSE_ROOT}/data/CampusSeq1`, respectively.

2. We have processed the camera parameters to our formats and you can download them from this repository. They lie in `${POSE_ROOT}/data/Shelf/` and `${POSE_ROOT}/data/CampusSeq1/`,  respectively.

3. Due to the limited and incomplete annotations of the two datasets, we don't train our model using this dataset. Instead, we directly use the 2D pose estimator trained on COCO, and use independent 3D human poses from the Panoptic dataset to train our 3D model. It lies in `${POSE_ROOT}/data/panoptic_training_pose.pkl`. See our paper for more details.

4. For testing, we first estimate 2D poses and generate 2D heatmaps for these two datasets in this repository.  The predicted poses can also download from the repository. They lie in `${POSE_ROOT}/data/Shelf/` and `${POSE_ROOT}/data/CampusSeq1/`,  respectively. You can also use the models trained on COCO dataset (like HigherHRNet) to generate 2D heatmaps directly.

The directory tree should look like this:
```
${POSE_ROOT}
|-- data
    |-- Shelf
    |   |-- Camera0
    |   |-- ...
    |   |-- Camera4
    |   |-- actorsGT.mat
    |   |-- calibration_shelf.json
    |   |-- pred_shelf_maskrcnn_hrnet_coco.pkl
    |-- CampusSeq1
    |   |-- Camera0
    |   |-- Camera1
    |   |-- Camera2
    |   |-- actorsGT.mat
    |   |-- calibration_campus.json
    |   |-- pred_campus_maskrcnn_hrnet_coco.pkl
    |-- panoptic_training_pose.pkl
```


### CMU Panoptic dataset
1. Download the dataset by following the instructions in [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) and extract them under `${POSE_ROOT}/data/panoptic_toolbox/data`.
- You can only download those sequences you need. You can also just download a subset of camera views by specifying the number of views (HD_Video_Number) and changing the camera order in `./scripts/getData.sh`. The sequences and camera views used in our project can be obtained from our paper.
- Note that we only use HD videos,  calibration data, and 3D Body Keypoint in the codes. You can comment out other irrelevant codes such as downloading 3D Face data in `./scripts/getData.sh`.
2. Download the pretrained backbone model from [pretrained backbone](https://1drv.ms/u/s!AjX41AtnTHeTjn3H9PGSLcbSC0bl?e=cw7SQg) and place it here: `${POSE_ROOT}/models/pose_resnet50_panoptic.pth.tar` (ResNet-50 pretrained on COCO dataset and finetuned jointly on Panoptic dataset and MPII).

The directory tree should look like this:
```
${POSE_ROOT}
|-- models
|   |-- pose_resnet50_panoptic.pth.tar
|-- data
    |-- panoptic-toolbox
        |-- data
            |-- 16060224_haggling1
            |   |-- hdImgs
            |   |-- hdvideos
            |   |-- hdPose3d_stage1_coco19
            |   |-- calibration_160224_haggling1.json
            |-- 160226_haggling1  
            |-- ...
```

## Training
### CMU Panoptic dataset

Train and validate on the five selected camera views. You can specify the GPU devices and batch size per GPU  in the config file. We trained our models on two GPUs.
```
python run/train_3d.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5.yaml
```
### Shelf/Campus datasets
```
python run/train_3d.py --cfg configs/shelf/prn64_cpn80x80x20.yaml
python run/train_3d.py --cfg configs/campus/prn64_cpn80x80x20.yaml
```

## Evaluation
### CMU Panoptic dataset

Evaluate the models. It will print evaluation results to the screen./
```
python test/evaluate.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5.yaml
```
### Shelf/Campus datasets

It will print the PCP results to the screen.
```
python test/evaluate.py --cfg configs/shelf/prn64_cpn80x80x20.yaml
python test/evaluate.py --cfg configs/campus/prn64_cpn80x80x20.yaml
```
