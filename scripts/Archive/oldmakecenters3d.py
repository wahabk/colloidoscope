def make_random_centers_3d(canvas_size, n, zoom, min_dist):
	'''
	Generate random centers of particles

	This is a place holder for bringing in simulated particle trajectories from dynamo
	'''
	canvas_size = [int(c/zoom) for c in canvas_size]
	min_dist = min_dist/zoom
	z = random.randint(0, canvas_size[0])
	y = random.randint(0, canvas_size[1])
	x = random.randint(0, canvas_size[2])
	centers = [(z,y,x)] # make first particle
	for i in range(n):
		too_close = True
		while too_close:
			z = random.randint(0, canvas_size[0])
			y = random.randint(0, canvas_size[1])
			x = random.randint(0, canvas_size[2])
			centers.append((z,y,x))
			distances = spatial.distance.pdist(centers)
			if all(i > min_dist for i in distances):
				too_close = False
				break
			else:
				centers.pop() # get rid of last element if too close
	return centers

def draw_sphere(canvas, center, r):
	cz, cy, cx = center
	for i in range(canvas.shape[0]):
			for j in range(canvas.shape[1]):
				for k in range(canvas.shape[2]):
					if (i - cz)**2 + (j - cy)**2 + (k - cx)**2 <= r**2:
						canvas[i,j,k] = 255
	return canvas

def draw_multiple_spheres(canvas, centers, r):
	for center in centers:
		cz, cy, cx = center
		for i in range(canvas.shape[0]):
				for j in range(canvas.shape[1]):
					for k in range(canvas.shape[2]):
						if (i - cz)**2 + (j - cy)**2 + (k - cx)**2 <= r**2:
							canvas[i,j,k] = 255
	return canvas