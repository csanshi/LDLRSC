import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.io as sio 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from munkres import Munkres
import os
from sklearn import cluster
import time
from utils import load_data
import yaml


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def get_encoder(input_image_shape, kernel_size, num_hidden):
	"""
	return: (x, x.shape)
	"""
	inputs = layers.Input(input_image_shape)
	x = inputs
	for i in range(len(kernel_size)):
		x = layers.Conv2D(num_hidden[i], kernel_size[i], activation='relu', strides=2, padding='same')(x)
	return keras.Model(inputs, x, name='encoder')

def get_decoder(input_shape, kernel_size, num_hidden):
	inputs = layers.Input(input_shape)
	num_hidden = num_hidden[-2::-1]
	num_hidden.append(1)
	kernel_size = kernel_size[-1::-1]
	x = inputs
	for i in range(len(kernel_size)):
		x = layers.Conv2DTranspose(num_hidden[i], kernel_size[i], activation='relu', strides=2, padding='same')(x)
	return keras.Model(inputs, x, name='decoder')


class ConvAE_pre(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(ConvAE_pre, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.total_loss_tracker
        ]

    def call(self, inputs):
        z_from_encoder = self.encoder(inputs)
        reconstruction = self.decoder(z_from_encoder)

		# reconstruction_loss = tf.reduce_sum(tf.pow(data - reconstruction, 2))
        return reconstruction

    def train_step(self, data):
		# 1.design kinds of losses
        with tf.GradientTape() as tape:
            z_from_encoder = self.encoder(data) # (N, 6, 6, 30)
            reconstruction = self.decoder(z_from_encoder) # (N, 48, 48, 1)
			
            reconstruction_loss = tf.reduce_sum(tf.pow(data - reconstruction, 2))
            total_loss = reconstruction_loss
        
        # 2.compute gradient & gradient descent
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		
		# 3.update all kinds of loss 
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
			"loss": self.total_loss_tracker.result(),
			"reconstruction_loss": self.reconstruction_loss_tracker.result(),
		}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load config

# args = yaml.load(open("EYaleB_config.yaml", 'r'))
args = yaml.load(open("ORL_config.yaml", 'r'))
# args = yaml.load(open("COIL20_config.yaml", 'r'))
# args = yaml.load(open("COIL100_config.yaml", 'r'))
# args = yaml.load(open("UMIST_config.yaml", 'r'))
# args = yaml.load(open("MNIST_config.yaml", 'r'))





num_class = args['dataset']['num_class']
num_per_class =  args['dataset']['num_per_class']
num_images = num_class * num_per_class
data_path = args['dataset']['data_path']


kernel_size = args['model']['kernel_size']
num_hidden = args['model']['num_hidden']
input_shape = args['model']['input_shape'] # [height, width]
input_shape_for_decoder = args['model']['input_shape_for_decoder']
pretrained_model = args['model']['pretrained_model']

num_epoch_pre = args['training']['num_epoch_pre']


if __name__ == '__main__':
	# load data
	(images, labels) = load_data(data_path, shape=input_shape) # ndarray
	images = tf.cast(images, dtype=tf.float32) # / 255

	images = tf.expand_dims(images, axis=-1) # below: 'images' is Tensor 
	if os.path.basename(data_path) == 'YaleBCrop025.mat':
		input_shape = [48, 48]
		images = tf.image.resize(images, input_shape) # resize to this specific size to fit Conv and Deconv
	if os.path.basename(data_path) == 'mnist1000.mat':
		input_shape = [32, 32]
		images = tf.image.resize(images, input_shape)

	convAE_pre = ConvAE_pre(encoder=get_encoder(input_shape+[1], kernel_size, num_hidden), 
							decoder=get_decoder(input_shape_for_decoder,  kernel_size, num_hidden))	# [6, 6, 30] [4, 4, 5]
	convAE_pre.compile(optimizer=keras.optimizers.Adam())

	# encoder=get_encoder(input_shape+[1], kernel_size, num_hidden)
	# decoder=get_decoder(input_shape_for_decoder,  kernel_size, num_hidden)
	# encoder.summary()
	# decoder.summary()
	
	print(images.shape)


	print('----------------------------pre_train: begin--------------------------')
	begin = time.time()

	convAE_pre.fit(images, epochs=num_epoch_pre, batch_size=num_images, verbose=0, shuffle=False)
	convAE_pre.save_weights(pretrained_model)

	print('Time used: %.1f' % (time.time() - begin))
	print('----------------------------pre_train: end----------------------------')