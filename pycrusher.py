import cppimport.import_hook
import crusher
import numpy as np 


def gen_rand_centers(volfrac, diameter = 10):
	centers = crusher.pycrush(volfrac)
	centers = (centers - np.min(centers))*diameter
	return centers

if __name__ == '__main__':
	centers = gen_rand_centers(0.35)
	print(centers)
	print(centers.shape, np.nanmin(centers), np.nanmax(centers))



# x = np.nan_to_num(x)

# import pdb; pdb.set_trace()

