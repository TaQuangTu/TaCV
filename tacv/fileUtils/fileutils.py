import os
import json

IMAGE_EXT = [".png", ".PNG", ".jpeg", ".JPEG", ".jpg", ".JPG"]


def get_all_files(dir, recursive=True, exts=None):
    all_files = []
    comps = os.listdir(dir)
    for comp in comps:
        full_path = os.path.join(dir, comp)
        file_ext = os.path.splitext(full_path)[1]
        if os.path.isdir(full_path):
            if recursive:
                all_files += get_all_files(full_path, recursive, exts)
            continue
        if exts == [] or exts is None:
            all_files.append(full_path)
        elif os.path.isfile(full_path) and file_ext in exts:
            all_files.append(full_path)
    return all_files


def save_json(file_name, json_data, mode="w+"):
    with open(file_name, mode) as f:
        json.dump(json_data, f)
        return True
    return False


def load_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)
    return None
