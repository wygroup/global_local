import networkx as nx
import numpy as np
import cmath
from matplotlib import pyplot as plt

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element

import os
import pickle
import warnings

from get_features import features

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

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
add_Ns = ['0N', '1N', '2N', '3N']

class utils:

	@staticmethod
	def draw_graph(g):
		#
		# draw a graph
		#
		nx.draw_networkx(g, pos=nx.spring_layout(g))
		plt.show()
	
	@staticmethod
	def load_graphs(filename):
		#
		# Load list of graph
		#
		with open(filename,'rb') as f:
			G = pickle.load(f)
		return G

	@staticmethod
	def save_graphs(G, filename):
		#
		# Save list of graph
		#
		with open(filename,'wb') as f:
			pickle.dump(G,f)

	@staticmethod
	def check_graphs(G):
		#
		# Delete graph which is not connected
		# G is a list of graph
		#
		for i,g in enumerate(G):
			if not nx.is_connected(g):
				#draw_graph(g)
				print(i, g.name)
				del G[i]
		print('checked graphs: ' + str(len(G)))

class Graphs:

	feature = np.array([])

	def atomic_distance(self, i, j):
		#
		# i,j are pymatgen Site objects
		#
		return (Element(i.specie).atomic_radius + Element(j.specie).atomic_radius)*1.2
	
	def add_polar_coordinates(self, g, struct):
		#
		# add polar coordinates of atom, metal atom is original point
		# add z coordinate of atom
		#
		original_coord = struct.sites[-1].coords
		polar_coord = (0, 0)
		for i, site in enumerate(struct.sites):
			site_coord = site.coords
			g.add_node(i, height = site_coord[-1])
			tmp = site_coord[:-1] - original_coord[:-1]
			tmp = complex(tmp[0], tmp[1])
			g.add_node(i, polar = cmath.polar(tmp))

	def add_features(self, g, struct):
		#
		# Add features of metal to graph
		#
		for i, site in enumerate(struct.sites):
			element = Element(site.specie)

			## feature is a string containing different attributes seperated by underline
			feature = str(element.Z) + ' ' + str(element.X) + ' ' + \
						str(element.row) + ' ' + str(element.group) + ' ' + \
						str(element.atomic_radius_calculated)
			#print(feature)
			g.add_node(i, feature=feature)

		# add additional feature to central metal
		feature_2 = feature + ' ' + str(features.d_metals[str(site.specie)]) + ' ' + \
					 str(features.IPs[str(site.specie)]) + ' ' + str(features.EAs[str(site.specie)]) + \
					 ' ' + str(features.Hs[str(site.specie)]) + ' ' + str(features.Ls[str(site.specie)]) + \
					' ' + str(features.ECs[str(site.specie)]) + ' ' + str(features.DCs[str(site.specie)])
		g.add_node(i, feature_2=feature_2)
	
	def generate_graph(self, mesh, add_N, s, e):
		#
		## create graph from POSCAR
		#
		## read structure information from POSCAR
		struct = Poscar.from_file('POSCAR', check_for_POTCAR=True, read_velocities=False).structure

		## initialize graph
		g = nx.Graph()
		g.name = mesh + ' ' + add_N + ' ' + s + ' ' + e

		## create features and add it to nodes
		self.add_features(g, struct)

		# polar coordinate
		self.add_polar_coordinates(g, struct)

		## create edges and add atom distance as feature to edges
		for i, site_1 in enumerate(struct.sites):
			for j, site_2 in enumerate(struct.sites):
				if site_1.distance(site_2) < self.atomic_distance(site_1, site_2) and i < j:
					g.add_edge(i, j, length=site_1.distance(site_2))
		return g

	def generate_graphs(self):
		#
		# return G, which is list of graphs
		#
		G = []
		for mesh in meshs:
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
						os.chdir(e)
						print(mesh,add_N,s,e)

						## generate graph for one structure
						g = self.generate_graph(mesh, add_N, s, e)

						## add all graphs to array G
						G.append(g)
						os.chdir('..')
					os.chdir('..')
				os.chdir('..')
			os.chdir('..')
		print('total graphs: ' + str(len(G)))
		return G

	def append_graphs(self):
		#
		# return G, which is list of graphs
		#
		# load existed graphs
		G = utils.load_graphs('graph.pkl')

		for mesh in ['42']:
			os.chdir(mesh)
			for add_N in ['2N', '3N']:
				os.chdir(add_N)
				for s in ['SV','SV_1N','SV_2N','SV_3N']:
					if not os.path.exists(s):
						continue
					os.chdir(s)
					for e in elements:
						os.chdir(e)
						print(mesh,add_N,s,e)

						## generate graph for one structure
						g = self.generate_graph(mesh, add_N, s, e)

						## add all graphs to array G
						G.append(g)
						os.chdir('..')
					os.chdir('..')
				os.chdir('..')
			os.chdir('..')
		print('total graphs: ' + str(len(G)))
		return G

	def CG(self, g, index, depth=3):
		#
		# Recursively add feature using graph representation,
		# 
		## sum all nodes if feature added here
		self.feature += np.array(g.nodes(data=True)[index]['feature'].split(' '), dtype=float)
		#print(depth, self.feature)

		## depth minus 1 after each loop
		if depth == 0:
			return self.feature
		depth -= 1

		## find all neighbouring node index
		adj_index = list(g.adj[index].keys())

		## Do not add metal atom more than once
		if g.number_of_nodes() - 1 in adj_index:
			adj_index.remove(g.number_of_nodes() - 1)
		#print(adj_index)
		for i in adj_index:
			CG(g, index=i, depth=depth)
			## sum nodes except root if feature added here
			#self.feature += np.array(G.nodes(data=True)[i]['feature'].split(' '), dtype=float)
			#print(depth, self.feature)

	@staticmethod
	def get_shell(g, depth = 3):
		#
		## return list containing shell information
		#
		# initialize current shell with metal atom index
		shell_index_current = np.array([g.number_of_nodes() - 1])
		shell_index_former = np.array([])
		# add feature of metal atom
		feature = np.array(g.nodes(data=True)[shell_index_current[0]]['feature_2'].split(' '), dtype=float)

		for _ in range(depth):
			# shell_index_former contains index of former shells
			shell_index_former = np.append(shell_index_former, shell_index_current)
			# shell_index_later contains index of later shell
			shell_index_later = np.array([])
			# concatenate index of neighbouring node at current shell
			for i in shell_index_current:
				shell_index_later = np.append(shell_index_later, list(g.adj[i].keys()))

			# delete repeated nodes, current shell moves to next depth
			shell_index_current = np.array([i for i in shell_index_later if i not in shell_index_former])
			#print(shell_index_current)
			# sum shell feature
			shell_feature = np.zeros((5,))
			for i in shell_index_current:
				shell_feature += np.array(g.nodes(data=True)[i]['feature'].split(' '), dtype=float)
			feature = np.append(feature, shell_feature / len(shell_index_current))

		return feature

	def update_graph(self, g):
		#
		# Update feature of all nodes in graph for one loop
		#
		g_new = g.copy()

		## update feature value of all nodes in graph
		for i in range(g.number_of_nodes()):
			fea_main = np.array(g.nodes(data=True)[i]['feature'].split(' '), dtype=float)
			fea_adj = np.zeros(np.shape(fea_main))
			num_adj = len(list(g.adj[i].keys()))

			## average adjacent nodes
			for j in list(g.adj[i].keys()):
				fea_adj += np.array(g.nodes(data=True)[j]['feature'].split(' '), dtype=float)
			fea_main = (fea_main + fea_adj) / (num_adj + 1)

			## update node
			g_new.nodes[i]['feature'] = ' '.join(np.array(fea_main, dtype=str))
		return g_new

	@staticmethod
	def update_feature(g, depth = 3):
		#
		# depth is times of iteration
		#
		for _ in range(depth):
			g = self.update_graph(g)
		index_metal = g.number_of_nodes() - 1
		fea_metal = np.array(g.nodes(data=True)[index_metal]['feature'].split(' '), dtype=float)
		# add additional feature of metal
		fea_metal = np.append(fea_metal, 
						np.array(g.nodes(data=True)[index_metal]['feature_2'].split(' '), dtype=float))
		return fea_metal

if __name__ == '__main__':
	# generate graphs
	#Gs = Graphs()
	#G = Gs.generate_graphs()
	#utils.save_graphs(G, 'graphs.pkl')

	## append graphs
	#Gs = Graphs()
	#G = Gs.append_graphs()
	#utils.save_graphs(G, 'graph.pkl')

	## check graphs
	G = utils.load_graphs('graphs.pkl')
	#utils.check_graphs(G)
	Gs = Graphs()
	f = Gs.get_shell(G[0])
	#utils.draw_graph(G[0])
