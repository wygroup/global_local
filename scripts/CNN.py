import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, GlobalAveragePooling2D, \
	Flatten, Dense, Activation, BatchNormalization, Dropout, concatenate, Add


def wide_deep(Input_1, Input_2):
	x_1 = Lenet5(Input_1)
	# x_1 = VGG_16(Input_1)
	# x_1 = resnet(Input_1)
	# Output = x_1
	# Output = Input_2
	Output = concatenate([x_1, Input_2])
	Output = Dense(2000)(Output)
	Output = Activation(tf.nn.relu)(Output)
	Output = Dropout(0.2)(Output)
	Output = Dense(200)(Output)
	Output = Activation(tf.nn.relu)(Output)
	Output = Dropout(0.2)(Output)
	Output = Dense(1)(Output)
	return Output


def Lenet5(Input):
	# Input shape (32,32)
	x = Conv2D(6, 5, 1, 'valid')(Input)
	x = MaxPool2D(2, 2, 'valid')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)

	x = Conv2D(16, 5, 1, 'valid')(x)
	x = MaxPool2D(2, 2, 'valid')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)

	x = Conv2D(120, 5, 1, 'valid')(x)

	x = Flatten()(x)

	x = Dense(84)(x)
	x = Activation(tf.nn.relu)(x)
	x = Dropout(0.2)(x)
	# x = Dense(1)(x)
	return x


def VGG_16(Input):
	# Input shape (32,32)
	x = Conv2D(64, 3, 1, 'same')(Input)
	x = Conv2D(64, 3, 1, 'same')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)
	x = MaxPool2D(2, 2, 'valid')(x)

	x = Conv2D(128, 3, 1, 'same')(x)
	x = Conv2D(128, 3, 1, 'same')(x)
	x = Conv2D(128, 3, 1, 'same')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)
	x = MaxPool2D(2, 2, 'valid')(x)

	x = Conv2D(256, 3, 1, 'same')(x)
	x = Conv2D(256, 3, 1, 'same')(x)
	x = Conv2D(256, 3, 1, 'same')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)
	x = MaxPool2D(2, 2, 'valid')(x)

	x = Conv2D(512, 3, 1, 'same')(x)
	x = Conv2D(512, 3, 1, 'same')(x)
	x = Conv2D(512, 3, 1, 'same')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)
	x = MaxPool2D(2, 2, 'valid')(x)

	x = Conv2D(512, 3, 1, 'same')(x)
	x = Conv2D(512, 3, 1, 'same')(x)
	x = Conv2D(512, 3, 1, 'same')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)
	x = MaxPool2D(2, 2, 'valid')(x)

	x = Flatten()(x)

	x = Dense(2000)(x)
	x = Activation(tf.nn.relu)(x)
	x = Dropout(0.2)(x)

	x = Dense(2000)(x)
	x = Activation(tf.nn.relu)(x)
	x = Dropout(0.2)(x)

	x = Dense(84)(x)
	x = Activation(tf.nn.relu)(x)
	x = Dropout(0.2)(x)

	return x


class MyModel(tf.keras.Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = Conv2D(6, 5, 1, 'valid', input_shape=(32, 32, 6))
		self.conv2 = Conv2D(16, 5, 1, 'valid')
		self.conv3 = Conv2D(120, 5, 1, 'valid')
		self.pool = MaxPool2D(2, 2, 'valid')
		self.BN = BatchNormalization()
		self.AC = Activation(tf.nn.relu)
		self.flatten = Flatten()
		self.dense1 = Dense(200)
		self.dense2 = Dense(2000)
		self.dense3 = Dense(1)

	def call(self, inputs):
		input_1 = inputs[0]
		input_2 = inputs[1]
		x = self.conv1(input_1)
		x = self.pool(x)
		x = self.BN(x)
		x = self.AC(x)
		x = self.conv2(x)
		x = self.pool(x)
		x = self.BN(x)
		x = self.AC(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)

		x = concatenate([x, input_2])
		x = self.dense2(x)
		x = self.AC(x)
		x = self.dense3(x)
		return x


def res_net_block(input_data, filters, conv_size):
	# CNN层
	x = Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
	x = BatchNormalization()(x)
	x = Conv2D(filters, conv_size, activation=None, padding='same')(x)
	# 第二层没有激活函数
	x = BatchNormalization()(x)
	# 两个张量相加
	x = Add()([x, input_data])
	# 对相加的结果使用ReLU激活
	x = Activation('relu')(x)
	# 返回结果
	return x


def resnet(Input):
	x = Conv2D(32, 3, activation='relu')(Input)
	x = Conv2D(64, 3, activation='relu')(x)
	x = MaxPool2D(3)(x)
	num_res_net_blocks = 10
	for i in range(num_res_net_blocks):
		x = res_net_block(x, 64, 3)
	# 添加一个CNN层
	x = Conv2D(64, 3, activation='relu')(x)
	# 全局平均池化GAP层
	x = GlobalAveragePooling2D()(x)
	# 几个密集分类层
	x = Dense(256, activation='relu')(x)
	# 退出层
	x = Dropout(0.5)(x)
	x = Dense(1, activation='relu')(x)
	return x
