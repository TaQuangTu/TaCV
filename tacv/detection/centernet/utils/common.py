import json
import os
import shutil


def recreate_dir(dir, cancel_if_exist=True):
    if os.path.exists(dir):
        if cancel_if_exist:
            return
        else:
            shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

