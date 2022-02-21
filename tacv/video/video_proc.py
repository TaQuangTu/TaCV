import os
import cv2
from ..fileUtils import get_all_files, IMAGE_EXT


def video2images(video_path, output_dir, exist_ok=False, image_ext="jpg", verbose=True, **kwargs):
    if not os.path.exists(video_path):
        if verbose:
            print(f"video {video_path} does not exist")
        return False
    if not os.path.exists(output_dir):
        if verbose:
            print(f"Making directory {output_dir}")
        os.makedirs(output_dir)
    else:
        if exist_ok:
            if verbose:
                print(f"Removing all files in {output_dir}")
            os.remove(output_dir)
            os.makedirs(output_dir)
        else:
            if verbose:
                print(
                    f"Output dir {output_dir} exists. pass True to exist_ok param or try removing all files in {output_dir} and re-run.")
    count = 0
    vidcap = cv2.VideoCapture(video_path)

    while True:
        success, image = vidcap.read()
        if success:
            path_to_save = os.path.join(output_dir, str(count) + f".{image_ext}")
            if verbose:
                print(f"Saving {path_to_save}")
            cv2.imwrite(path_to_save, image)
            count = count + 1
        else:
            break
    print(f"Extracted {count} frames")
    vidcap.release()
    return True


def images2video(image_dir, video_path, fps=24,image_ext: str=None, sort=False, **kwargs):
    if not os.path.exists(image_dir):
        print(f"Image dir {image_dir} does not exist")
        return False
    video_dir = os.path.dirname(video_path)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    if image_ext is None:
        exts = None
    else:
        exts = [image_ext]
    frame_paths = get_all_files(image_dir, recursive=False, exts=exts)
    if sort:
        frame_paths = sorted(frame_paths)
    assert len(frame_paths) > 0, f"There is no image in {image_dir}"
    test_image = cv2.imread(frame_paths[0])
    shape = test_image.shape
    frame_size = (shape[1],shape[0])
    video_writer = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*"MJPG"),fps,frame_size)


    try:
        for frame_path in frame_paths:
            image = cv2.imread(frame_path)
            video_writer.write(image)
    except Exception as e:
        print(e)
        if os.path.exists(video_path):
            os.remove(video_path)
        return False
    print(f"Saving video successfully at {video_path}")
    video_writer.release()
    return True
