import numpy as np
import colloidoscope
import matplotlib.pyplot as plt

dc = colloidoscope.DeepColloid()
array = dc.read_tif('examples/Data/emily.tiff')
array = dc.crop3d(array, roiSize=(128,128,128))
print(array.shape)
df, positions, label = dc.detect(array, diameter=13, weights_path='output/weights/attention_unet_202206.pt', debug=True, device="cpu")
print(df)

x,y = dc.get_gr(positions, 50,50)
plt.plot(x,y,)
plt.savefig("output/test/gr.png")