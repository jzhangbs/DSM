# Deep Stereo Matchability

## Environment
- GPU mem >= 16G (training, batch size 4 on single GPU)
- GPU mem >= 7G (testing, batch size 1, full resolution image on single GPU)
- CUDA >= 10.0
- Python >= 3.6
    - pytorch >= 1.0
    - opencv-python

## Data Preparation
1. Download Sceneflow dataset from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html.
   Please download the RGB images (cleanpass) and Disparity of three subsets. Then extract the files to the corresponding
   subfolder. e.g. For flyingthings3d, extract RGB images and disparity and you will get two folder named disparity and
   frames_cleanpass. Put them in `<data_root>/flyingthings3d/`.
2. Download KITTI 2012 from http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo,
   KITTI 2015 from http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo.
   For each dataset, extract and you will get two folders named training and testing. Put them in `<data_root>/kitti/201x/unzip/`.

## Training
First pretrain the model on Sceneflow.
```
$ python train.py \
  --data_root <data_root> \
  --dataset d,m,f \
  --base unet \
  --lr 1e-3,.5e-3,.25e-3,.125e-3 \
  --boundaries .625,.75,.875 \
  --epoch 16 \
  --batch_size 16 \
  --job_name <sceneflow_job_name> \
  --save_dir <save_dir>
```
The model will be stored in `<save_dir>/<sceneflow_job_name>/`.

Then finetune the model on KITTI
```
$ python train.py \
  --data_root <data_root> \
  --dataset k15 \
  --base unet \
  --lr 1e-3,1e-4,1e-5 \
  --boundaries .33,.67 \
  --epoch 600 \
  --batch_size 16 \
  --load_path <save_dir>/<sceneflow_job_name> \
  --reset_step \
  --job_name <kitti_job_name> \
  --save_dir <save_dir>
```
The model will be stored in `<save_dir>/<kitti_job_name>/`.

## Testing
To evaluate the model on flyingthings3d:
```
$ python val.py \
  --data_root <data_root> \
  --dataset f \
  --base unet \
  --load_path <save_dir>/<sceneflow_job_name> \
```

And to produce the disparity of KITTI test set:
```
$ python val.py \
  --data_root <data_root> \
  --dataset k15 \
  --base unet \
  --load_path <save_dir>/<kitti_job_name> \
  --write_result \
  --result_dir <result_dir>
```
The outputs will be stored in <result_dir>. Note that the program will report dummy EPE and precision because there is
no ground truth.

## Pretrained Model
We provide the pretrained model of the architecture with UNet base model. Extract the model and use `model/unet_sceneflow`
as the load path.
