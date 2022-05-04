# Colloidoscope!

A project to attempt tracking colloids using confocal and deep learning. Please note this is work-in-progress.

Colloidoscope is essentially a U-net trained on simulated data. Before using colloidoscope I would use TrackPy or nplocate. This is designed to work on very small colloids (~200nm) with a very bad point spread function, that is anisotropic and has high z blur.

Pretrained weights and hyperparameter training history available upon request.

# Installation

Quickest (not preferred)

```
pip3 install git+https://github.com/wahabk/colloidoscope
```


Option 1, make sure you use a virtual environment:

```
cd <installation_dir>
git clone https://github.com/wahabk/colloidoscope
# activate your virtual environment, make sure you use python 3.8
python3.8 -m pip install .
```

Option 2, To get going with conda (most robust and includes hoomdblue):

```
cd <installation_dir>
git clone https://github.com/wahabk/colloidoscope
conda env create -f environment.yaml # this will create an env using python 3.8 named 'colloids'
conda activate colloids
python3.8 -m pip install .
```

# Usage: Quick Start

I have no documentation for this project but I will endeavor to write nice docstrings. I have provided custom jupyter notebooks in ```examples/```.

Colloidoscope centers around the class ```DeepColloid```. This contains all the functions you will need.

First import and read your data:

```Python
import colloidoscope as cd

dc = cd.DeepColloid()
path = 'path/to/image.tif'
image = dc.read_tif(path)
# If you have a lif file Colloidoscope wraps the explore_lif Reader class
Reader = dc.ExploreLifReader(args, kwargs)
```

Then simply use ```dc.detect()``` to track your image, this will return a pandas DataFrame just like TrackPy.

```Python
df = dc.detect(image, weights_path='path/to/model_weights.pt')
```

## Model limitations
- This only accepts np arrays that are at least [32,128,128] in size (z,x,y)
- Note the dimension order is (z,x,y), this is important since the model has only been trained on simulations with z blur in this order
- This has been tested on particles with size 4 - 12 pixel radius

## Useful things
- All available average precision functions only work with bounding boxes. Colloidoscope has a function ```dc.average_precision(true_positions, predicted_precisions)``` that will find AP based on positions alone - since the distance is more important for this project than the IOU.
- You can use / adapt dc.simulate() if you want to quickly draw simulated particles in 3D. 

# Dependencies

 - Python 3.8
 - pathlib2 
 - trackpy 
 - pytorch 
 - numpy 
 - numba 
 - matplotlib 
 - h5py 
 - napari 
 - scipy 
 - scikit-learn 
 - scikit-image 
 - torchio 
 - neptune 
 - hoomdblue (only for simulations)

# Dev 
## Simulations

If you want to use colloidoscopes positions simulation you will need to install [hoomdblue](https://github.com/glotzerlab/hoomd-blue). This is included if you used Option 2 when installing but please install on your own if you used pip (Option 1). I found the easiest way is to use conda:

```conda install -c conda-forge hoomd -y```

## Docker

I provide a docker image to make gpu training easier

Step 1 : Build image, this will build an image named ```colloids```

```
docker build . --tag=colloids 
docker build . --tag=colloids --network=host # if you're on vpn
```

Step 2 : Create container in interactive mode, this will start a shell inside the container

```
docker run -it \
	-v <data_dir>:<data_dir> \ # mount data directory as volume
	-v <repo_dir>:/colloidoscope \ # mount git repo directory as volume for interactive access
	--gpus all \ # allow the container to use all gpus
	--network=host \ # if you're on vpn
	colloids \ 
```

Note:
If you want to launch the container on a custom hard drive use:

```sudo dockerd --data-root <custom_dir>```

## Contributions

Thanks to yushi yang for contributing the position simulations and helping me with the entire project in general.

This project wraps explore_lif which is written by Mathieu Leocmach and C. Ybert.
