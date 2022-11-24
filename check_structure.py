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

warnings.filterwarnings("ignore")

substrates = ['SV', 'SV_1N', 'SV_2N', 'SV_3N',
              'DV', 'DV_1N', 'DV_2N_1', 'DV_2N_2', 'DV_3N', 'DV_4N',
              'HV', 'HV_1N', 'HV_2N_1', 'HV_2N_2', 'HV_3N', 'HV_4N']
# substrates = ['SV','SV_1N','SV_2N','SV_3N']


elements = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']

row_1 = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
row_2 = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag']
row_3 = ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']

meshs = ['42', '24', '22', '42_2']

add_Ns = ['0N', '1N', '2N', '3N']


def atomic_distance(i, j):
	#
	# i,j: pymatgen PeriodicSite objects
	#
	return (Element(i.specie).atomic_radius + Element(j.specie).atomic_radius) * 1.2


def logs(filename, log):
	#
	# record log file
	#
	with open('../../../../' + filename, 'a') as f:
		f.write(log + '\n')


if __name__ == '__main__':

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
					print('now processing: ', mesh, add_N, s, e)
					# create Structure from POSCAR
					struct = Poscar.from_file('POSCAR', check_for_POTCAR=True, read_velocities=False).structure

					count = 0
					log = ''
					for i, site in enumerate(struct.sites):
						if struct.get_distance(i, len(struct.sites) - 1) < atomic_distance(site, struct.sites[-1]):
							count += 1
					count -= 1  # 减去自身
					log = '	'.join([mesh, add_N, s, e]) + '	' + str(count)
					if count < 3:
						logs('structure_all.log', log)
						print(log)

					os.chdir('../')
				os.chdir('../')
			os.chdir('../')
		os.chdir('../')
