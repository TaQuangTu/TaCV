import sys
from ..video import video2images, images2video


def cli_images2video():
    args = sys.argv[1:]
    images2video(*args)


def cli_video2images():
    args = sys.argv[1:]
    video2images(*args)
