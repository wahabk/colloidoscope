import numpy as np
import colloidoscope

dc = colloidoscope.DeepColloid()
array = dc.read_tif('examples/Data/emily.tiff')
print(array.shape)
dc.detect(array, weights_path='output/weights/attention_unet_202206.pt')