from setuptools import setup
import setuptools
import os

def read_version_int(version_file_path):
    if not os.path.exists(version_file_path):
        return None
    with open(version_file_path,"r") as version_file:
        text = version_file.readlines()[0]
        version_file.close()
        batches = text.split(".")
        batches = [int(batch) for batch in batches]
        return batches

def increase_version_by_one(version_file_path):
    batches = read_version_int(version_file_path)
    if batches is None:
        return
    batch1, batch2, batch3 = batches
    batch3 += 1
    if batch3 >= 10:
        batch3 = batch3 % 10
        batch2 += 1
        if batch2 >= 10:
            batch1 = batch1 + 1
    version_file = open(version_file_path,"w+")
    new_ver = str(batch1)+"."+str(batch2)+"."+str(batch3)
    version_file.write(new_ver+"\n")
    version_file.close()
    return new_ver

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as f:
    lines = f.readlines()
    required_pkgs = [item.strip() for item in lines]

#version_file = "tacv/resources/version.txt"
new_ver = "1.1.4" #increase_version_by_one(version_file)
print(f"Building {new_ver}")
setup(
    name='tacv',
    version=new_ver,
    packages=setuptools.find_packages(where="."),
    url='https://github.com/TaQuangTu/TaCV',
    license='LICENSE',
    author='TaQuangTu',
    install_requires=required_pkgs,
    author_email='taquangtu132@gmail.com',
    description='A mini package for daily tasks',
    long_description_content_type="text/markdown",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
