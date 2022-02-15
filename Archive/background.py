import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import napari
import numpy as np
from random import randint

def make_background(canvas_size, octaves, brightness):
    noise = PerlinNoise(octaves=2)
    zpix, xpix, ypix = canvas_size
    pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)] 
    pic = [pic for _ in range(zpix)]

    pic = np.array(pic)
    pic = pic-pic.min()
    pic = pic/pic.max()
    pic = pic*10

    return pic

canvas_size = (32, 128, 128)
pic = make_background(canvas_size, 4, randint(0,20))

print(pic.max(), pic.min(), pic.mean())

napari.view_image(pic)
napari.run()
