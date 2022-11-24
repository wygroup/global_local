import networkx as nx
import numpy as np
import cmath
from matplotlib import pyplot as plt
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element
import os
import datetime
import pickle
import pandas as pd
import warnings
from get_features import features
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Flatten
from tf_geometric.nn import gcn


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
		# draw a graph
		nx.draw_networkx(g, pos=nx.spring_layout(g))
		# plt.show()
	
	@staticmethod
	def load_graphs(filename):
		# Load list of graph
		with open(filename, 'rb') as f:
			G = pickle.load(f)
		return G
	
	@staticmethod
	def save_graphs(G, filename):
		# Save list of graph
		with open(filename, 'wb') as f:
			pickle.dump(G, f)
	
	@staticmethod
	def check_graphs(G):
		# Delete graph which is not connected
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
	def add_polar_coordinates(g, struct):
		# add polar coordinates of atom, metal atom is original point
		# add z coordinate of atom
		original_coord = struct.sites[-1].coords
		for i, site in enumerate(struct.sites):
			site_coord = site.coords
			g.add_node(i, height=site_coord[-1])
			tmp = site_coord[:-1] - original_coord[:-1]
			tmp = complex(tmp[0], tmp[1])
			g.add_node(i, polar=cmath.polar(tmp))
	
	@staticmethod
	def add_features(g, struct):
		#
		# Add features of metal to graph
		#
		for i, site in enumerate(struct.sites):
			element = Element(site.specie)

			# feature is a string containing different attributes seperated by underline
			feature = str(element.Z) + ' ' + str(element.X) + ' ' \
			          + str(element.row) + ' ' + str(element.group) + ' ' \
			          + str(element.atomic_radius_calculated)
			# print(feature)
			g.add_node(i, feature=feature)

		# add additional feature to central metal
		site = struct.sites[-1]
		index = len(struct.sites) - 1
		feature = g.nodes[index]['feature']
		feature_2 = feature + ' ' + str(features.d_metals[str(site.specie)]) + ' ' \
		            + str(features.IPs[str(site.specie)]) + ' ' \
		            + str(features.EAs[str(site.specie)]) + ' ' \
		            + str(features.Hs[str(site.specie)]) + ' ' \
		            + str(features.Ls[str(site.specie)]) + ' ' \
		            + str(features.ECs[str(site.specie)]) + ' ' \
		            + str(features.DCs[str(site.specie)])
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


def generate_graph(mesh, add_N, s, e):
	# create graph from POSCAR
	# read structure information from POSCAR
	struct = Poscar.from_file('POSCAR', check_for_POTCAR=True, read_velocities=False).structure
	
	# initialize graph
	g = nx.Graph()
	g.name = mesh + ' ' + add_N + ' ' + s + ' ' + e
	
	# create features and add it to nodes
	Utils.add_features(g, struct)
	
	# polar coordinate
	Utils.add_polar_coordinates(g, struct)
	
	# create edges and add atom distance as feature to edges
	for i, site_1 in enumerate(struct.sites):
		for j, site_2 in enumerate(struct.sites):
			if site_1.distance(site_2) < Utils.atomic_distance(site_1, site_2) and i < j:
				g.add_edge(i, j, length=site_1.distance(site_2))
	return g


def generate_graphs():
	#
	# return G, which is list of graphs
	#
	os.chdir('../')
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
					print(mesh, add_N, s, e)
					
					# generate graph for one structure
					g = generate_graph(mesh, add_N, s, e)
					nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False,
					                 node_color=["gray" for _ in range(len(g.nodes) - 1)] + ["darkcyan"])
					plt.show()
					exit()
					
					# add all graphs to array G
					G.append(g)
					os.chdir('..')
				os.chdir('..')
			os.chdir('..')
		os.chdir('..')
	print('total graphs: ' + str(len(G)))
	return G


def DAD_X(graph):
	# calculate D * A
	A = nx.adjacency_matrix(graph).todense()
	A = A + np.eye(len(A))
	D = np.diag([np.sum(i) for i in A])
	DAD = np.matmul(np.linalg.inv(D), A)
	DAD = np.array(DAD)
	
	X = []
	for n in graph.nodes:
		feature = graph.nodes[n]['feature']
		feature = feature.split()
		X.append(feature)
	return DAD, np.asfarray(X, dtype='float')


class MyLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs, DAD):
		super(MyLayer, self).__init__()
		self.num_outputs = num_outputs
		self.DAD = DAD

	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
		                              shape=[int(input_shape[-1]), self.num_outputs],
		                              trainable=True)

	# DAD * X * W
	def call(self, X, **kwargs):
		return tf.matmul(tf.matmul(self.DAD, X), self.kernel)


class MyDense(tf.keras.layers.Layer):
	def __init__(self):
		super(MyDense, self).__init__()

	def call(self, inputs, **kwargs):
		return inputs[:, -1, :]


def dat_to_db(filename) -> pd.DataFrame:
	#
	# Convert dat file to pandas.DataFrame
	#
	print('converting dat file to database ...')
	vector = ['mesh', 'add_N', 'sub', 'metal', 'E_ads_H']
	with open(filename, 'r') as f:
		data = f.readlines()[1:]
		name = []
		E = []
		for d in data:
			name.append(d.strip().split('	')[:-1])
			E.append(d.strip().split('	')[-1])
	E = np.array(E, dtype=float)

	# create pandas DataFrame
	db = pd.DataFrame(name, columns=vector[:-1])
	db.insert(len(db.columns), 'E_ads_H', E)
	return db


def data_clean(*filenames):
	#
	# exclude unexpected structure and adsorption configuration
	#
	print('data cleaning ...')
	name = []
	for filename in filenames:
		with open(filename, 'r') as f:
			data = f.readlines()
		for l in data:
			name.append(l.split('	')[:-1])
	return name


if __name__ == '__main__':
	G = generate_graphs()
	print(len(G))
	with open("C:/Users/chenyuzhuo/Desktop/ML/graphs_old.pkl", "wb") as f:
		pickle.dump(G, f)
	exit()

	tf.keras.backend.set_floatx('float32')
	tf.compat.v1.disable_eager_execution()

	clean_name = data_clean('../structure.log', '../ads_H/structure.log')

	with open("../graphs.pkl", 'rb') as f:
		graphs = pickle.load(f)
	print("loading graphs ...")
	Xs, DADs = [], []
	for graph in graphs:
		DAD, X = DAD_X(graph)
		Xs.append(X)
		DADs.append(DAD)
	Xs = np.array(Xs)
	DADs = np.array(DADs)

	y = []
	print("loading y ...")
	db = dat_to_db('../All.dat')
	for i, j in db.iterrows():
		# data clean
		# if [j['mesh'], j['add_N'], j['sub'], j['metal']] in clean_name:
			# continue
		if 'V' in j['sub']:
			y.append(j['E_ads_H'])
	y = np.array(y)

	print("shape of Xs, DADs, y: ", Xs.shape, DADs.shape, y.shape)
	# exit()

	# data standard
	# scalar = preprocessing.StandardScaler().fit(X)
	# scalar = preprocessing.MinMaxScaler().fit(X)
	# X = scalar.transform(X)

	inputs_X = tf.keras.Input(shape=(None, 5))
	inputs_DAD = tf.keras.Input(shape=(None, None))

	# build model
	outputs = MyLayer(20, inputs_DAD)(inputs_X)
	outputs = Activation(tf.nn.relu)(outputs)
	outputs = MyLayer(15, inputs_DAD)(outputs)
	outputs = Activation(tf.nn.relu)(outputs)
	outputs = MyLayer(10, inputs_DAD)(outputs)
	outputs = MyDense()(outputs)
	# x = tf.keras.layers.Average()([*x])
	# x = tf.keras.layers.Maximum()([*x])
	outputs = Dense(1)(outputs)

	model = tf.keras.Model(inputs=[inputs_X, inputs_DAD], outputs=outputs)
	model.summary()

	model.compile(optimizer='Adam', loss='mse', metrics=['mae', 'mse'])

	rundir = "C:/Users/chenyuzhuo/Desktop/ML/scripts/logs/fit/MPNN/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	model.fit([Xs, DADs], y, epochs=100, verbose=2, batch_size=1,
	          callbacks=[
		          tf.keras.callbacks.TensorBoard(log_dir=rundir, histogram_freq=1),
	          ]
	          )
	print(model.predict([Xs, DADs]))



