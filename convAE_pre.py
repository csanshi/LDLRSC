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

def load_data():
	"""
	return: 
		images: ndarray (n, height, width, 1)
		labels: ndarray (n, 1)
	"""
	images = sio.loadmat('Data/YaleBCrop025.mat')['Y'] # (2016, 64, 38)
	images = np.transpose(images) # (38, 64, 2016)
	(num_class, num_image_per_class, dim) = images.shape # (38, 64, 2016)
	images = np.reshape(images, (num_class*num_image_per_class, dim)) # (2432, 2016)
	images = np.reshape(images, (images.shape[0], 42, 48)) # (2432, 42, 48)
	images = np.transpose(images, (0, 2, 1)) # (2432, 48, 42)
	images = tf.cast(images, dtype=tf.float32) / 255
	labels = np.zeros(images.shape[0], np.int8)
	for _class in range(1, num_class+1):
		labels[_class*num_image_per_class:(_class+1)*num_image_per_class] = _class
	return (images, labels)

def get_encoder(input_image_shape):
	"""
	return: (x, x.shape)
	"""
	inputs = layers.Input(input_image_shape)
	x = layers.Conv2D(10, 5, activation='relu', strides=2, padding='same')(inputs)
	x = layers.Conv2D(20, 3, activation='relu', strides=2, padding='same')(x)
	x = layers.Conv2D(30, 3, activation='relu', strides=2, padding='same')(x)
	return keras.Model(inputs, x, name='encoder')

def get_decoder(input_shape):
	inputs = layers.Input(input_shape)
	x = layers.Conv2DTranspose(20, 3, activation='relu', strides=2, padding='same')(inputs)
	x = layers.Conv2DTranspose(10, 3, activation='relu', strides=2, padding='same')(x)
	x = layers.Conv2DTranspose(1, 5, activation='relu', strides=2, padding='same')(x)
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



if __name__ == '__main__':
	begin = time.time()
    # load data
	(images, labels) = load_data() # ndarray
	images = tf.expand_dims(images, axis=-1) # below: 'images' is Tensor 
	images = tf.image.resize(images, [48, 48])
	# images = tf.cast(images, dtype=tf.float32) / 255

	# setting up
	EPOCH_PRE_TRAIN = 20
	CHECKPOINT_PATH_PRE_TRAIN = 'checkpoint/pre_train/ck.ckpt'

	batch_size = images.shape[0]
	convAE_pre = ConvAE_pre(encoder=get_encoder(images[0].shape), 
							decoder=get_decoder([6, 6, 30]))
	convAE_pre.compile(optimizer=keras.optimizers.Adam())

	encoder=get_encoder(images[0].shape)
	decoder=get_decoder([6, 6, 30])
	encoder.summary()
	decoder.summary()
	# # if len(os.listdir(os.path.dirname(CHECKPOINT_PATH_PRE_TRAIN))) == 0:
	# print('----------------------------pre_train: begin--------------------------')
	# convAE_pre.fit(images, epochs=EPOCH_PRE_TRAIN, batch_size=batch_size, verbose=2, shuffle=False)
	# convAE_pre.save_weights(CHECKPOINT_PATH_PRE_TRAIN)
	# # convAE_pre.summary()
	# print('----------------------------pre_train: end----------------------------')
	# print('Time used: %.1f' % (time.time() - begin))