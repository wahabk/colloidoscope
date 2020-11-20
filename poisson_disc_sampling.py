# from https://github.com/KamilKrol5/bridson-algorithm
# THANK YOU KAMIL!!!

from itertools import product
from typing import List
import numpy as np

Point = np.ndarray
Shape = np.ndarray
ListOfPoints = List[np.ndarray]
NDimArray = np.ndarray


def _is_sample_valid(sample: Point,
                     sample_region_size: Shape,
                     grid: NDimArray,
                     cell_size: float,
                     radius: float,
                     points: ListOfPoints) -> bool:
    if any(sample < 0) or any(sample > sample_region_size):
        return False

    # sample's cell index in the grid
    cell_index = (sample // cell_size).astype(int)

    start = ((max(coordinate, 0)
              for coordinate in cell_index - 2))
    end = ((min(coordinate, length - 1)
            for coordinate, length
            in zip(cell_index + 2, grid.shape)))
    ranges = [list(range(start_, end_ + 1))
              for (start_, end_) in zip(start, end)]

    for index in product(*ranges):
        point_index = grid[index]
        if point_index != -1:
            distance = np.linalg.norm(sample - points[point_index])
            if distance < radius:
                return False
    return True


def _get_random_n_dim_vector(dimension: int, min_length: float, max_length: float) -> Point:
    # need n-1 angles, all in range [0; PI] except the last one which is in range [0; 2*PI]
    random_angles = np.random.rand(dimension - 1) * np.pi
    random_angles[-1] *= 2
    length = np.random.uniform(min_length, max_length)
    vector = np.empty(dimension)
    for i in range(dimension):
        x_i = length * np.product(np.sin(random_angles[:i]))
        if i != dimension - 1:
            x_i *= np.cos(random_angles[i])
        vector[i] = x_i

    return vector


def poisson_disc_sampling(radius: float, sample_domain_size: Shape, sample_rejection_threshold=30) -> ListOfPoints:
    """
    Returns a list of random points from the sampling domain such that the distance between any two points
    is at least the radius. The sampling is done using Bridson algorithm. This implementation supports sampling
    in n-dimensional spaces.

        Parameters
        ----------
        radius : float
            The minimum distance between points.

        sample_domain_size : ndarray
            An array with dimensions of sampling domain. For example if [X, Y] array is provided,
            the sampling space has 2 dimensions and is a X x Y rectangle. If [X, Y, Z] is provided the
            sampling space is X x Y x Z cuboid. In general n-dimensional domains are supported.

        sample_rejection_threshold : int, optional
            The number which defines how much samples from the neighbourhood of a given point is tested.
            Default is 30.

        Returns
        -------
        points : List[ndarray]
            A list of samples (points).

        Notes
        -----
        If the sample_rejection_threshold parameter is too low, the points may not be distributed evenly.
        There may be even some large free spaces. The default is set to be 30 - it value recommended
        by the algorithm's author - Robert Bridson. More information in the paper linked below.
        https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
           """
    dimension = sample_domain_size.shape[0]

    cell_size: float = radius / np.sqrt(dimension)
    grid_shape = np.ceil(sample_domain_size / cell_size).astype(int)

    # contains indexes of points in the 'points' list
    grid = np.full(shape=grid_shape, fill_value=-1)
    points: ListOfPoints = []
    active_points: ListOfPoints = []

    initial_sample = np.random.rand(dimension) * sample_domain_size / 2
    active_points.append(initial_sample)

    while len(active_points) > 0:
        random_index = np.random.choice(len(active_points))
        random_sample = active_points[random_index]

        for _ in range(sample_rejection_threshold):
            sample_candidate = random_sample + _get_random_n_dim_vector(dimension, radius, 2 * radius)
            if _is_sample_valid(sample_candidate, sample_domain_size, grid, cell_size, radius, points):
                points.append(sample_candidate)
                active_points.append(sample_candidate)
                candidate_grid_index = tuple(int(x) for x in (sample_candidate // cell_size))
                grid[candidate_grid_index] = len(points) - 1
                break
        else:
            active_points.pop(random_index)

    return points
