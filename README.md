# TACV - A mini package for daily tasks

## Installation
```bash
pip install tacv
```

## Examples
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
images2video(image_dir,video_path)
```
#### Extract images from a video
```python
from tacv.video import video2images
video_path = "tacv_test.mp4" #path to video to be extracted to images
image_dir = "my_images" #directory to save the extracted images
video2images(video_path,image_dir)
```
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