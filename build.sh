rm -rf dist
rm -rf tacv.egg-info
rm -rf build
python -m build
twine upload dist/*