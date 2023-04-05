# What it is
## Author: Yushi Yang

This folder contains script that generate hard sphere configurations with different polydispersities.
It used the simulation package [hoomd-blue](https://hoomd-blue.readthedocs.io/en/stable/package-hpmc.html).
For this project, the version number of the hoomd-blue is 2.9.7, being the *stable* version in 2021.
It is important to use the *stable* version rather than the *latest* one (3.0.x), since there are a lot of API changes.


## How to use it

### Generate simulation results

You are expected to use

```sh
python3 generate_hs_conf.py
```

To generate the hard sphere configurations.
The properties of the system, like the overall volume, the number of particles, and the polydispersity, can be changed by editing the variables in the `generate_hs_conf.py` file.

### Load the result as numpy array

The configuration is written into a `gsd` file. These files can be loaded to software `Ovito` for visualisation.

We need another package, [pygsd](https://gsd.readthedocs.io/en/stable/python-module-gsd.pygsd.html), to access the data stored in the `gsd` files. I provide a pedagogic script (`read_gsd.py`), demonstrating the way to load the information as a reference.
