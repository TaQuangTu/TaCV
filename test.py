from tacv.fileUtils.fileutils import get_all_files
from tacv.fileUtils.fileutils import save_json, load_json
from tacv.visual.draw2d import draw_points
import cv2

if __name__ == "__main__":
    # save
    file_list = get_all_files("../tupkgs")

    json_file = "myfile.json"
    json_data = {"name": "Ta", "age": 100}
    # save json
    save_json(json_file, json_data)
    # load json
    json_data = load_json(json_file)

    # draw 2d points onto an image
    image = cv2.imread("myimage.jpg")
    points = [(18,19),(55,55),(102,22),(66,22)]
    draw_points(image,points,circular=True,color=(0,255,0),thickness=2)