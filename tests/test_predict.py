import numpy as np
import colloidoscope
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np

dc = colloidoscope.DeepColloid()
array = dc.read_tif('examples/Data/katherine.tiff')
# array = dc.crop3d(array, roiSize=(128,128,128))
print(array.shape, array.dtype)
df, positions, label = dc.detect(array, diameter=7, 
                                weights_path='output/weights/attention_unet_202206.pt', 
                                debug=True, device="cpu",
                                post_processing="tp", batch_size=1)
# array = np.array(array, dtype="uint8")
# df =  dc.run_trackpy(array, diameter=13, preprocess=False, max_iterations=1)
# print(df)

x,y = dc.get_gr(positions, 50,50)
plt.plot(x,y,)
plt.savefig(f"output/test/gr.png")

# dc.view(array,  label=label,  positions=positions)
