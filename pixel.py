import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import warnings
import pandas as pd

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter

warnings.filterwarnings("ignore")

substrates = ['SV','SV_1N','SV_2N','SV_3N',
			'DV','DV_1N','DV_2N_1','DV_2N_2','DV_3N','DV_4N',
			'HV','HV_1N','HV_2N_1','HV_2N_2','HV_3N','HV_4N']
#substrates = ['SV','SV_1N','SV_2N','SV_3N']


elements = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
			'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 
			'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']
row_1 = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
row_2 = ['Y', 'Zr', 'Nb','Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag']
row_3 = ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']

meshs = ['42', '24', '22', '42_2']
#meshs = ['42', '42_2']
add_Ns = ['0N', '1N', '2N', '3N']

d_metals = {'Sc':1,'Ti':2,'V':3,'Cr':5,'Mn':5,'Fe':6,'Co':7,'Ni':8,'Cu':10,'Zn':10,
			'Y':1,'Zr':2,'Nb':4,'Mo':5,'Tc':5,'Ru':7,'Rh':8,'Pd':10,'Ag':10, 'Cd':10,
			'Hf':2,'Ta':3,'W':4,'Re':5,'Os':6,'Ir':7,'Pt':9,'Au':10}

def atomic_distance(i,j):
	#
	# i,j: pymatgen PeriodicSite objects, covalent length between two species
	#
	return (Element(i.specie).atomic_radius + Element(j.specie).atomic_radius)*1.2

def visualize_data(X):
	#
	# X: list of images, [batch, height, width]
	#
	for _i, i in enumerate(X):
		plt.subplot(2, 5, _i+1)
		plt.imshow(i, cmap='viridis')
	plt.show()

def expand_cell(struct, size, grid):
	#
	# expand cell according to size and grid of pixel image
	#
	desired_length = size[0] * grid + 1.0
	scaling_matrix = [1,1,1]
	for i, l in enumerate(struct.lattice.abc[:2]):
		# find scaling factor and make it even
		factor = desired_length // l + 1
		if factor % 2 == 0:
			factor += 1
		scaling_matrix[i] = np.int(factor)
	#print(scaling_matrix)
	struct.make_supercell(scaling_matrix=scaling_matrix)

def find_central_atom(struct):
	#
	# struct: pymatgen Structure object, find cetral metal atoms after expanding cell
	#
	sites = []
	for site in struct.sites:
		if site.specie.symbol in elements:
			sites.append(site.coords)
	sites = np.array(sites)
	coords = np.zeros((3,))
	coords[0] = (sites[:,0].max() + sites[:,0].min()) / 2
	coords[1] = (sites[:,1].max() + sites[:,1].min()) / 2
	coords[2] = sites[:,2][0]
	return coords

def find_min_z(struct):
	#
	# struct: pymatgen Structure object, find lowest atom in structure
	#
	min_z = 100.
	for site in struct.sites:
		if site.z < min_z:
			min_z = site.z
	return min_z

def add_gaussian_blur(image, sigma = 1.5):
	#
	# image format: (height, width, channels)
	#
	# decomposite channels dimension
	decomp_image = [image[:,:,i] for i in range(image.shape[-1])]
	for index, i in enumerate(decomp_image):
		decomp_image[index] = gaussian_filter(i, sigma = sigma)
	# add channels dimension
	decomp_image = [np.expand_dims(i, axis = -1) for i in decomp_image]
	# concatenate channels dimension
	return tf.concat(decomp_image, axis = -1)

def pixel(size=(128, 128, 6), grid=0.25, sigma = 1.5):
	#
	# create pixel image from POSCAR
	#
	# read structure from Poscar
	POSCAR = Poscar.from_file('POSCAR', 
								check_for_POTCAR=True,
								read_velocities=False,
								)
	struct = POSCAR.structure
	
	# expand cell 
	expand_cell(struct, size[:2], grid)
	
	# find central metal atom coords in expanded cell and compress it to be 2-dimensional
	central_atom_coords = find_central_atom(struct)[:2]
	image_length = size[0] * grid
	base_point = central_atom_coords - image_length / 2
	
	# initialize pixel image
	image = np.zeros(size)

	# find minimal z coordinate
	min_z = find_min_z(struct)

	for site in struct.sites:
		if site.x < base_point[0] or site.x >= base_point[0] + image_length:
			continue
		if site.y < base_point[1] or site.y >= base_point[1] + image_length:
			continue
		index_x = (site.x - base_point[0]) // grid 
		index_y = (site.y - base_point[1]) // grid 

		# set z coords as one channel
		image[np.int(index_x)][np.int(index_y)][0] = site.z - min_z
		element = Element(site.specie)
		# atomic number
		image[np.int(index_x)][np.int(index_y)][1] = element.Z
		# electronegativity
		image[np.int(index_x)][np.int(index_y)][2] = element.X
		# row
		image[np.int(index_x)][np.int(index_y)][3] = element.row
		# group
		image[np.int(index_x)][np.int(index_y)][4] = element.group
		# atomic_radius_calculated
		image[np.int(index_x)][np.int(index_y)][5] = element.atomic_radius_calculated

	# Gaussian filter
	image = add_gaussian_blur(image, sigma = sigma)


	# translate image
	vectors = np.array([[16,16],[12,12],[8,8],[4,4],
				[12,16],[8,16],[4,16],
				[16,12],[16,8],[16,4]])
	vectors = vectors * 2
	crop_images = np.array([tf.image.crop_to_bounding_box(image, i, j, 64, 64) for i, j in vectors])
	#visualize_data(crop_images[:,:,:,1])
	#exit()

	# resize images
	crop_images = tf.image.resize(images=crop_images, size=(32,32), method='gaussian')
	#visualize_data(crop_images[:,:,:,1]) 

	# filp image
	#image = tf.image.flip_up_down(image)
	flip_images = tf.image.flip_left_right(crop_images)
	#exit()
	return np.append(crop_images, flip_images, axis=0)

if __name__ == '__main__':
	# P format: (batch, height, width, channel)
	images = []
	for mesh in meshs:
		if not os.path.exists(mesh):
			continue
		os.chdir(mesh)
		for add_N in add_Ns:
			if not os.path.exists(add_N):
				continue
			os.chdir(add_N)
			for s in substrates:
				if not os.path.exists(s):
					continue
				os.chdir(s)
				for e in elements:
					if not os.path.exists(e):
						continue
					os.chdir(e)
					print('now processing: ', mesh,add_N,s,e)
					# create image from POSCAR
					crop_images = pixel()
					for i in crop_images:
						images.append((mesh + ' ' + add_N + ' ' + s + ' ' + e, i))
					os.chdir('../')
				os.chdir('../')
			os.chdir('../')
		os.chdir('../')
	print(np.shape(images))

	with open('pixels_all_20.pkl','wb') as f:
		pickle.dump(np.array(images), f)