# TACV - A mini package for daily tasks

## Installation
```bash
pip install tacv
```

## Examples
### 2D Object Detection

For now, CenterNet supported. However, use it as prototype purpose only, there is no official benchmark on accuracy.
* First, create a config file for training/model config, see full config at `tacv/detection/sample_config.yml`.
```yaml
input_size: &input_size [ 224,448 ]
max_object: &max_obj 16
num_classes: &num_classes 5
train_config:
  gpus: 0 # 0 means CPU, N means using N available GPU(s) for training
  epoch: 600
  batch_size: 32
  shuffle: True
  num_workers: 4
  learning_rate: 0.0001
  lr_decay_milestones: [ 80,160 ]
  lr_decay_gamma: 0.5
  weight_decay: 0.01
  checkpoint_frequency: 1
  amp: True
  unfreeze_bbone_epoch: 200
  initial_denom_lr: 5
  loss_hm_reg_offset_weights: [ 1, 1, 0.1 ]
  callback:
    monitor: "val_loss"
    dirpath: "logs/exp_name_1"
    save_top_k: 20
    mode: "min"
val_config:
  batch_size: 1
  checkpoint: ""
model:
  num_classes: *num_classes
  backbone_layers: 18
  head_conv_channel: 64
  max_object: *max_obj
  input_shape: *input_size
```
* Second, create your own Dataset class that returns data as described in the `__getitem__()` method, see following example:
```python
from torch.utils.data import Dataset
import torch

class MockDataset(Dataset):
    def __init__(self, max_objs):
        self.max_objs = max_objs

    def __getitem__(self, item):
        image = torch.rand(3, 224, 448)  # Shape = (3, H, W)
        annos = torch.rand(self.max_objs, 5)  # Shape = (MaxObjs x 5) , each row presents for (x,y,w,h,class_id)
        masks = torch.zeros(
            self.max_objs)  # Shape = (MaxObjs,)  each value is False or True (1 indicates having object)
        masks[0:3] = True
        return {"image": image, "annos": annos, "masks": masks}

    def __len__(self):
        return 1000
```
* Init `CenterNetTrainer` and here we go
```python
from tacv.detection import CenterNetTrainer
from torch.utils.data import random_split

config_path = "tacv/detection/sample_config.yml"
dataset = MockDataset(max_objs=16)  # Replace with your custom dataset
train_set, val_set = random_split(dataset, [len(dataset)*0.9, len(dataset)-len(dataset)*0.9])

trainer = CenterNetTrainer(train_set, val_set, config_path)
trainer.train()
```
* Finally, do inference (TO BE UPDATED after coffee time).

### File utils
#### Get all file paths from a directory
```python
from tacv.fileUtils import get_all_files
file_paths = get_all_files("dir_name")
```
Returns a list of file absolute paths, for example
```
['./venvCondaTest/x86_64-conda_cos6-linux-gnu/bin/ld', './venvCondaTest/conda-meta/_libgcc_mutex-0.1-main.json', './venvCondaTest/conda-meta/xz-5.2.5-h7b6447c_0.json', './venvCondaTest/conda-meta/wheel-0.37.1-pyhd3eb1b0_0.json', './venvCondaTest/conda-meta/setuptools-58.0.4-py36h06a4308_0.json', './venvCondaTest/conda-meta/ca-certificates-2021.10.26-h06a4308_2.json', './venvCondaTest/conda-meta/readline-8.1.2-h7f8727e_1.json', './venvCondaTest/conda-meta/sqlite-3.37.2-hc218d9a_0.json', './venvCondaTest/conda-meta/libgcc-ng-9.3.0-h5101ec6_17.json', './venvCondaTest/conda-meta/ncurses-6.3-h7f8727e_2.json']
```
#### Save/load json data to/from file
```python
from tacv.fileUtils import save_json,load_json

json_file = "myfile.json"
json_data = {"name":"Ta","age":100}
# save json
save_json(json_file,json_data)
# load json
json_data = load_json(json_file)
```
### Visual
#### Draw 2D points onto an image
```python
import cv2
from tacv.visual import draw_points
image = cv2.imread("myimage.jpg")
points = [(18,19),(55,55),(102,22),(66,22)]
draw_points(image,points,circular=True,color=(0,255,0),thickness=2)
cv2.imwrite("new_image.jpg",image)
```
### Video
#### Synthesize a video from images
```python
from tacv.video import images2video
image_dir = "my_images" #directory containing images in the same format
video_path = "tacv_test.mp4" #path to save the synthesized video
# common use case
images2video(image_dir,video_path,fps=24, image_ext = None, sort=False)
```
Parameters:
* `fps`: default = 24
* `image_ext`: a string, specify image extension to synthesize the video, for example (`jpg`, `png`,...). If it is `None`. All images will be grabbed. Default is `None`.
* `sort`: `True` or `False`. Indicate if the images should be sorted by name before synthesizing the video. Default is `True`.
#### Extract images from a video
```python
from tacv.video import video2images
video_path = "tacv_test.mp4" #path to video to be extracted to images
image_dir = "my_images" #directory to save the extracted images
video2images(video_path,image_dir,exist_ok=False, image_ext="jpg", verbose=True)
```
Parameters:
* `exist_ok`: default is False. If `image_dir` already contains images and this flag is `False`. The process will be cancel, otherwise it continues.
* `image_ext`: a string, specify image extension, for example (`jpg`, `png`,...). If it is `None`. All images will be grabbed. Default is `None`.
* `verbose`: `True` or `False`. Set it to `True` to view the extracting process. Default is `True`.
### Geometry
#### Calculate 2D IOU of two polygons
```python
from tacv.geometry import iou_2d
polygon_1 = [[0,0],[10,10],[0,10]]
polygon_2 = [[0, 20], [10, 10], [0, 0]]
print(iou_2d(polygon_1,polygon_2))
```
### Command Line Interface

#### Synthesize a video from images
```bash
tacv_i2v image_dir video_path [optional: fps image_ext]
```
#### Extract images from a video
```bash
tacv_v2i video_path image_dir
```
### For more
* Visit args description in source code 
* Visit `test.py` file