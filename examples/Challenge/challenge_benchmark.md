# Benchmark

For a simple benchmark [TrackPy](https://github.com/soft-matter/trackpy) is a great place to start tracking spheres.

Trackpy uses a centroid finding algorithm and is usually the first port of call for tracking colloids.

```python
import trackpy as tp

diameter = x
array = # Read your image here

df = tp.locate(array, diameter=diameter)
```

The only argument that trackpy needs is the diameter (in pixels) of the particles.