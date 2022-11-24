import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import warnings
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter

warnings.filterwarnings("ignore")

substrates = ['SV', 'SV_1N', 'SV_2N', 'SV_3N',
			  'DV', 'DV_1N', 'DV_2N_1', 'DV_2N_2', 'DV_3N', 'DV_4N',
			  'HV', 'HV_1N', 'HV_2N_1', 'HV_2N_2', 'HV_3N', 'HV_4N']
# substrates = ['SV_3N']
elements = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
			'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
			'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']
# elements = ['Pt']
meshs = ['42', '24', '22', '42_2']
# meshs = ['22']
add_Ns = ['0N', '1N', '2N', '3N']
# add_Ns = ['3N']
d_metals = {'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 5, 'Mn': 5, 'Fe': 6, 'Co': 7, 'Ni': 8, 'Cu': 10, 'Zn': 10,
			'Y': 1, 'Zr': 2, 'Nb': 4, 'Mo': 5, 'Tc': 5, 'Ru': 7, 'Rh': 8, 'Pd': 10, 'Ag': 10, 'Cd': 10,
			'Hf': 2, 'Ta': 3, 'W': 4, 'Re': 5, 'Os': 6, 'Ir': 7, 'Pt': 9, 'Au': 10}


def atomic_distance(i, j):
	return (Element(i.specie).atomic_radius + Element(j.specie).atomic_radius) * 1.2


def visualize_data(X):
	"""
	X: [height, width, channel]
	"""
	for _i in range(X.shape[-1]):
		plt.subplot(2, 3, _i + 1)
		plt.imshow(X[:, :, _i], cmap='viridis')
	plt.show()


def expand_cell(struct, size, grid):
	desired_length = size[0] * grid + 1.0
	scaling_matrix = [1, 1, 1]
	for i, l in enumerate(struct.lattice.abc[:2]):
		# find scaling factor and make it even
		factor = desired_length // l + 1
		if factor % 2 == 0:
			factor += 1
		scaling_matrix[i] = np.int(factor)
	# print(scaling_matrix)
	struct.make_supercell(scaling_matrix=scaling_matrix)


def find_central_atom(struct):
	"""
	find central TM after cell expansion
	"""
	sites = []
	for site in struct.sites:
		if site.specie.symbol in elements:
			sites.append(site.coords)
	sites = np.array(sites)
	coords = np.zeros((3,))
	coords[0] = (sites[:, 0].max() + sites[:, 0].min()) / 2
	coords[1] = (sites[:, 1].max() + sites[:, 1].min()) / 2
	coords[2] = sites[:, 2][0]
	return coords


def find_min_z(struct):
	"""
	find lowest z coordination
	"""
	min_z = 100.
	for site in struct.sites:
		if site.z < min_z:
			min_z = site.z
	return min_z


def add_gaussian_blur(image, sigma=1.5):
	"""
	add Gaussian blur to every channel
	"""
	# decomposite channels dimension
	decomp_image = [image[:, :, i] for i in range(image.shape[-1])]
	for index, i in enumerate(decomp_image):
		decomp_image[index] = gaussian_filter(i, sigma=sigma)
	# add channels dimension
	decomp_image = [np.expand_dims(i, axis=-1) for i in decomp_image]
	# concatenate channels dimension
	return tf.concat(decomp_image, axis=-1).numpy()


def pixel(filepath, size=(128, 128, 6), grid=0.20, sigma=1.5):
	"""
	create pixel image from POSCAR
	"""
	struct = Poscar.from_file(os.path.join(filepath, 'POSCAR'), check_for_POTCAR=True, read_velocities=False).structure

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
		# atomic number
		element = Element(site.specie)
		image[np.int(index_x)][np.int(index_y)][1] = element.Z
		# image[np.int(index_x)][np.int(index_y)][1] = 2 if element.Z == 6 else element.Z
		# electronegativity
		image[np.int(index_x)][np.int(index_y)][2] = element.X
		# row
		image[np.int(index_x)][np.int(index_y)][3] = element.row
		# group
		image[np.int(index_x)][np.int(index_y)][4] = element.group
		# atomic_radius_calculated
		image[np.int(index_x)][np.int(index_y)][5] = element.atomic_radius_calculated

	# Gaussian filter
	image = add_gaussian_blur(image, sigma=sigma)
	# visualize_data([image[:, :, 1]])
	return image


def data_augmentation(images):
	"""
	input: images, [batch, height, width, channel]
	output: augmentation images, [batch * 20, height, width, channel]
	"""
	vectors = np.array([[16, 16], [12, 12], [8, 8], [4, 4],
						[12, 16], [8, 16], [4, 16],
						[16, 12], [16, 8], [16, 4]])
	vectors = vectors * 2

	print('data augmentation ...')
	images_new = []
	for name, image in images:
		print(name)
		# translate image
		crop_images = np.array([tf.image.crop_to_bounding_box(image, i, j, 64, 64).numpy() for i, j in vectors])
		# flip image
		flip_images = tf.image.flip_left_right(crop_images).numpy()
		# flip_images = tf.image.flip_up_down(crop_images)
		# resize image
		crop_images = tf.image.resize(images=crop_images, size=(32, 32), method='gaussian').numpy()
		flip_images = tf.image.resize(images=flip_images, size=(32, 32), method='gaussian').numpy()
		aug_images = np.append(crop_images, flip_images, axis=0)

		for aug_image in aug_images:
			images_new.append((name, aug_image))

	return np.array(images_new)


if __name__ == '__main__':
	# format: (batch, height, width, channel)
	root_dir = os.getcwd()

	images_origin = []
	for mesh in meshs:
		file_path_1 = os.path.join(root_dir, mesh)
		if not os.path.exists(file_path_1):
			continue
		for add_N in add_Ns:
			file_path_2 = os.path.join(file_path_1, add_N)
			if not os.path.exists(file_path_2):
				continue
			for sub in substrates:
				file_path_3 = os.path.join(file_path_2, sub)
				if not os.path.exists(file_path_3):
					continue
				for e in elements:
					file_path_4 = os.path.join(file_path_3, e)
					if not os.path.exists(file_path_4):
						continue
					print(f"now processing: {file_path_4}")
					# create image from POSCAR
					image = pixel(file_path_4)
					images_origin.append((mesh + ' ' + add_N + ' ' + sub + ' ' + e, image))
	images_origin = np.array(images_origin)
	print(images_origin.shape, images_origin[0][1].shape)
	images = data_augmentation(images_origin)
	print(images.shape, images[0][1].shape)

	with open(os.path.join(root_dir, "pixels.pkl"), 'wb') as f:
		joblib.dump(images, f)  # images is a list of tuple containing name `str` and pixel `np.ndarray`

	print("DONE")
