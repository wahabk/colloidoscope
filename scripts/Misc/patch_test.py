from patchify import patchify, unpatchify
import numpy as np

image = np.zeros((128,128,128))

patch_size = (64,64,64)

patches = patchify(image, patch_size, 64)

print(patches.shape)

recon = unpatchify(patches, image.shape)

print(recon.shape)
