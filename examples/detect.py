import colloidoscope as cd

if __name__ == "__main__":
    dc = cd.DeepColloid()

    model_weights_path = 'path/to/model_weights.pt'
    path = 'path/to/image.tif'
    image = dc.read_tif(path) 

    df = dc.detect(image, weights_path=model_weights_path, diameter=9)