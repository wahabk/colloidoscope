# Colloidoscope!

A project to attempt tracking colloids using confocal and deep learning.

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

# Installation

Option 1:

```
cd <installation_dir>
git clone https://github.com/wahabk/colloidoscope
# activate your virtual environment, make sure you use python 3.8
python3 -m pip install .
```

Option 2, To get going with conda (most robust and includes hoomdblue):

```
cd <installation_dir>
git clone https://github.com/wahabk/colloidoscope
conda env create -f environment.yaml # this will create an env using python 3.8 named 'colloids'
conda activate colloids
python3 -m pip install .
```

# Usage: Quick Start

I will provide custom jupyter notebooks in ```examples/```.

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
