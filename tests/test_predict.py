import numpy as np
import colloidoscope
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
from colloidoscope.train_utils import exclude_borders

dc = colloidoscope.DeepColloid()
array = dc.read_tif('examples/Data/jamesdecon.tiff')
array = dc.crop3d(array, roiSize=(256,256,256))
print(array.shape, array.dtype)
df, positions, label = dc.detect(array, diameter=15, 
                                weights_path='output/weights/attention_unet_202211.pt', 
                                patch_overlap=(16,16,16),
                                debug=True, run_on="cpu",
                                roiSize=(100,100,100), label_size=(96,96,96),
                                post_processing="log", batch_size=1)
print(df)
array = np.array(array, dtype="uint8")
tp_pred, df =  dc.run_trackpy(array, diameter=15)
print(df)

tp_pred = exclude_borders(tp_pred, (256,256,256), pad = 15)
positions = exclude_borders(positions, (256,256,256), pad = 15)

x, y = dc.get_gr(tp_pred, 100, 100)
plt.plot(x, y, label=f'tp n ={len(tp_pred)}', color='gray')
x, y = dc.get_gr(positions, 100, 100)
plt.plot(x, y, label=f'Unet n ={len(positions)}', color='red')
plt.savefig(f"output/test/gr.png")

# dc.view(array,  label=label,  positions=positions)
