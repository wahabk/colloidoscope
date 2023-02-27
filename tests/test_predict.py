import numpy as np
import colloidoscope
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
from colloidoscope.train_utils import exclude_borders
from scipy import ndimage
import tifffile

dc = colloidoscope.DeepColloid()
array = dc.read_tif('examples/Data/levke2.tiff')
print(array.shape, array.dtype, len(array.shape))
# array = dc.read_tif('/home/wahab/Data/HDD/Colloids/Real/Levke/stedImagesLevke/smallOld_pxSize_good_large_30nm_5frames_decon.tif')[0]
d = 15
# array = ndimage.zoom(array, 1.5)
weights_path = 'output/weights/attention_unet_202211.pt'
array = dc.crop3d(array, roiSize=(200,500,500))
# array = np.array([array, array])
print(array.shape, array.dtype, len(array.shape))
tifffile.imsave('examples/Data/levke.tiff', array)
exit()
df, positions, label = dc.detect(array, diameter=d, 
                                weights_path=weights_path, 
                                patch_overlap=(16,16,16),
                                debug=True, run_on="cpu",
                                roiSize=(64,64,64), label_size=(60,60,60),
                                post_processing="tp", batch_size=1)
print(df)
array = np.array(array, dtype="uint8")
tp_pred, df =  dc.run_trackpy(array, diameter=d)
print(df)

x, y = dc.get_gr(tp_pred, 100, 100)
plt.plot(x, y, label=f'tp n ={len(tp_pred)}', color='gray')
x, y = dc.get_gr(positions, 100, 100)
plt.plot(x, y, label=f'Unet n ={len(positions)}', color='red')
plt.savefig(f"output/test/gr.png")

print(positions.shape, tp_pred.shape)

dc.view(array,  label=label,  positions=positions)
