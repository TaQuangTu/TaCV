import os
import json

IMAGE_EXT = ["png", "PNG", "jpeg", "JPEG", "jpg", "JPG"]


def get_all_files(dir, recursive=True, exts=None):
    all_files = []
    comps = os.listdir(dir)
    for comp in comps:
        full_path = os.path.join(dir, comp)
        file_ext = os.path.splitext(full_path)[1]
        file_ext = file_ext[1:]  # ignore the dot
        if os.path.isdir(full_path):
            if recursive:
                all_files += get_all_files(full_path, recursive, exts)
            continue
        if exts == [] or exts is None:
            all_files.append(full_path)
        elif os.path.isfile(full_path) and file_ext in exts:
            all_files.append(full_path)
    return all_files


def get_file_name(file_path):
    """
    Get base name of a file from its absolute path or relative path, exclude the file extension. For example:
    base name of "/home/ubuntu/my_image.jpeg" is "my_image".
    :param file_path: path of a file
    :return: base name of the file
    """
    path, _ = os.path.splitext(file_path)
    return os.path.basename(path)


def get_file_extension(file_path):
    """
    Get file extension from file path
    :param file_path: any path
    :return: ext, return empty string ("") if the file does not have extension
    """
    return os.path.splitext(file_path)[1]


def save_json(file_name, json_data, mode="w+"):
    with open(file_name, mode) as f:
        json.dump(json_data, f)
        return True
    return False


def load_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)
    return None
