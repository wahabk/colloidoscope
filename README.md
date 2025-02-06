# Colloidoscope!

<a href="https://colab.research.google.com/github/wahabk/colloidoscope/blob/master/Colloidoscope_tutorial.ipynb"> 
<img src="https://colab.research.google.com/assets/colab-badge.svg"> </a>

<img src="examples/Challenge/figWebsiteLittle.png" alt="paddyfig1"/>

A project to track colloids using confocal and deep learning.

Colloidoscope is essentially a U-net trained on simulated data. This is designed to to improve tracking on very small colloids (~200nm) with a very bad point spread function, that is anisotropic and has high z blur.

Pretrained weights are included by default. Hyperparameter training history available upon request.

# Installation

```
pip3 install git+https://github.com/wahabk/colloidoscope
```


# Usage: Quick Start

I have provided custom jupyter notebooks in ```examples/```.

You can run colloidoscope in google colab: 
<a href="https://colab.research.google.com/github/wahabk/colloidoscope/blob/master/Colloidoscope_tutorial.ipynb"> 
<img src="https://colab.research.google.com/assets/colab-badge.svg"> </a>

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
df = dc.detect(image)
```

Please check `examples/` for scripts and jupyter notebooks.

## Useful things
- All available average precision functions only work with bounding boxes. Colloidoscope has a function ```dc.average_precision(true_positions, predicted_precisions)``` that will find AP based on positions alone - since the distance is more important for this project than the IOU.
- You can use / adapt dc.simulate() if you want to quickly draw simulated particles in 3D. 

# Dependencies

Check `setup.py`.

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

Thanks to [Yushi Yang @yangyushi](https://github.com/yangyushi) for contributing the position simulations and helping me with the entire project in general.

This project wraps explore_lif which is written by Mathieu Leocmach and C. Ybert.

## Citation

Incoming! 🚧
