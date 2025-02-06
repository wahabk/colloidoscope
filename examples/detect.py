import colloidoscope as cd

if __name__ == "__main__":
    dc = cd.DeepColloid()

    path = 'path/to/image.tif'
    image = dc.read_tif(path) 

    df = dc.detect(image, diameter=9)