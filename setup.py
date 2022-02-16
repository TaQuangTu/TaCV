from setuptools import setup
import setuptools

with open("README.md","r",encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='tacv',
    version='1.0.1',
    packages= setuptools.find_packages(where="."),
    url='https://github.com/TaQuangTu/TaCV',
    license='',
    author='TaQuangTu',
    author_email='taquangtu132@gmail.com',
    description='A mini package for daily tasks',
    long_description_content_type = "text/markdown",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
