import networkx as nx
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element
import os
import pickle
import warnings
from get_features import features


warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

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


class Utils:
	def __init__(self):
		super(Utils, self).__init__()
	
	@staticmethod
	def draw_graph(g):
		nx.draw_networkx(g, pos=nx.spring_layout(g))
		# plt.show()
	
	@staticmethod
	def load_graphs(filename):
		with open(filename, 'rb') as f:
			G = pickle.load(f)
		return G
	
	@staticmethod
	def save_graphs(G, filename):
		with open(filename, 'wb') as f:
			pickle.dump(G, f)
	
	@staticmethod
	def check_graphs(G):
		"""
		Delete graph which is not connected
		"""
		for i, g in enumerate(G):
			if not nx.is_connected(g):
				# draw_graph(g)
				print(i, g.name)
				del G[i]
		print('checked graphs: ' + str(len(G)))
	
	@staticmethod
	def atomic_distance(i, j):
		# i,j are pymatgen Site objects
		return (Element(i.specie).atomic_radius + Element(j.specie).atomic_radius) * 1.2

	@staticmethod
	def add_features(g, struct):
		"""
		Add features of metal to graph
		"""
		for i, site in enumerate(struct.sites):
			element = Element(site.specie)

			# feature is a string containing different attributes seperated by ' '
			feature = str(element.Z) + ' ' + str(element.X) + ' ' \
			          + str(element.row) + ' ' + str(element.group) + ' ' \
			          + str(element.atomic_radius_calculated)
			# print(feature)
			g.add_node(i, feature=feature)

		# add additional feature to TM
		site = struct.sites[-1]
		index = len(struct.sites) - 1
		feature = g.nodes[index]['feature']
		feature_2 = feature + ' ' + str(features.d_metals[str(site.specie)]) + ' ' \
		            + str(features.IPs[str(site.specie)]) + ' ' \
		            + str(features.EAs[str(site.specie)]) + ' ' \
		            + str(features.Hs[str(site.specie)]) + ' ' \
		            + str(features.Ls[str(site.specie)])
		g.add_node(index, feature_2=feature_2)

	@staticmethod
	def get_shells(graph: nx.Graph):
		feature = np.asfarray(graph.nodes[graph.number_of_nodes() - 1]['feature_2'].split())
		adj_nodes = list(list(graph.adjacency())[-1][-1].keys())
		if len(adj_nodes) != 0:
			fea = []
			for n in adj_nodes:
				fea.append(graph.nodes[n]['feature'].split())
			fea = np.asfarray(fea)
			fea = np.average(fea, axis=0)
		else:
			fea = np.zeros(5)
		return np.concatenate((feature, fea), axis=0)


def generate_graph(mesh, add_N, s, e, filepath):
	"""
	create graph from POSCAR and POTCAR using pymatgen
	"""
	# read structure information from POSCAR
	struct = Poscar.from_file(os.path.join(filepath, 'POSCAR'), check_for_POTCAR=True, read_velocities=False).structure
	
	# initialize graph
	g = nx.Graph()
	g.name = mesh + ' ' + add_N + ' ' + s + ' ' + e
	
	# create features and add it to nodes
	Utils.add_features(g, struct)
	
	# create edges and add atom distance as feature to edges
	for i, site_1 in enumerate(struct.sites):
		for j, site_2 in enumerate(struct.sites):
			if site_1.distance(site_2) < Utils.atomic_distance(site_1, site_2) and i < j:
				g.add_edge(i, j, length=site_1.distance(site_2))
	return g


def generate_graphs():
	"""
	return list of graphs G
	"""
	root_dir = os.getcwd()
	G = []
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

					# generate graph for one structure
					g = generate_graph(mesh, add_N, sub, e, file_path_4)
					G.append(g)
	print(f"total graphs: {len(G)}")
	return G


if __name__ == '__main__':
	G = generate_graphs()
	with open("./graphs.pkl", "wb") as f:
		pickle.dump(G, f)


