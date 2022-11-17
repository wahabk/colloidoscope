import numpy as np
import colloidoscope
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np

dc = colloidoscope.DeepColloid()
array = dc.read_tif('examples/Data/jamesdecon.tiff')
array = dc.crop3d(array, roiSize=(256,256,256))
print(array.shape, array.dtype)
df, positions, label = dc.detect(array, diameter=15, 
                                weights_path='output/weights/attention_unet_202206.pt', 
                                patch_overlap=(16,16,16),
                                debug=True, device="cpu",
                                post_processing="log", batch_size=1)
print(df)
array = np.array(array, dtype="uint8")
tp_pred, df =  dc.run_trackpy(array, diameter=15)
print(df)

x, y = dc.get_gr(tp_pred, 100, 100)
plt.plot(x, y, label=f'tp n ={len(tp_pred)}', color='gray')
x, y = dc.get_gr(positions, 100, 100)
plt.plot(x, y, label=f'Unet n ={len(positions)}', color='red')
plt.savefig(f"output/test/gr.png")

dc.view(array,  label=label,  positions=positions)
