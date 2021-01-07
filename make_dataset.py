from sim3d import simulate_img3d
from myhdf5 import read_hdf5, write_hdf5
from varname import nameof
from mainviewer import mainViewer


if __name__ == "__main__":
	canvas_size=(50,125,125)
	r = 10
	zoom = 0.75
	gauss = (7,3,3)
	min_dist = r+7
	k = 30
	dataset = 'test'

	for n in range(1,11):
		print(n)
		canvas, positions = simulate_img3d(canvas_size, r, min_dist, zoom, gauss, k=30)
		write_hdf5(dataset, canvas, positions, n=n)
		
	for n in range(1,11):
		canvas, positions = read_hdf5(dataset, n)
		mainViewer(canvas)
		print(len(positions), positions)