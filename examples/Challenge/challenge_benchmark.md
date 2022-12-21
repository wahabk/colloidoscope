# Benchmark

For a simple benchmark [TrackPy](https://github.com/soft-matter/trackpy) is a great place to start tracking spheres.

Trackpy uses a centroid finding algorithm and is usually the first port of call for tracking colloids.

```python
import trackpy as tp

diameter = x
array = # Read your image here

df = tp.locate(array, diameter=diameter)
```

The only argument that trackpy needs is the diameter (in pixels) of the particles - note this must be an odd number.

To get a numpy array use the following, note the dimension order zxy, this can be changed to whatever you're used to, but ensure you remain consistent and feed the dimension order to the read_y function.

```python
L = list(zip(df['z'], df['y'], df['x']))
tp_predictions = np.array(L, dtype='float32')
```