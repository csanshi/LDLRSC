import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import time
import yaml
import utils

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

######################################### ConvAE
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


class ConvAE(keras.Model):
	def __init__(self, encoder, decoder, batch_size, lambdas, gama, init_C=1.0e-8, **kwargs):
		super(ConvAE, self).__init__(**kwargs)
		
		self.lambda_rec, self.lambda_reg, self.lambda_exp, self.lambda_lap2 = lambdas
		self.gama = gama

		self.encoder = encoder
		self.decoder = decoder

		self.Coef = tf.Variable(init_C*tf.ones([batch_size, batch_size], tf.float32), name='Coef')

		self.selfexpression_loss_tracker = keras.metrics.Mean(name='selfexpression_loss')
		self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
		self.regularizer_loss_tracker = keras.metrics.Mean(name='regularizer_loss')
		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.laplace2_loss_tracker = keras.metrics.Mean(name="laplace2_loss")

	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.selfexpression_loss_tracker,
			self.reconstruction_loss_tracker,
			self.regularizer_loss_tracker,
            self.laplace2_loss_tracker
		]

	def call(self, inputs):
		z_from_encoder = self.encoder(inputs)
		z = layers.Flatten()(z_from_encoder)
		z_c = tf.matmul(self.Coef, z) # (batch_size, batch_size)*(batch_size, 1080)
		z_c_to_decoder = tf.reshape(z_c, z_from_encoder.shape) # (batch_size, 6, 6, 30)
		reconstruction = self.decoder(z_c_to_decoder)
		return reconstruction

	def train_step(self, data):
		
		with tf.GradientTape() as tape:
			z_from_encoder = self.encoder(data)
			z = layers.Flatten()(z_from_encoder)
			z_c = tf.matmul(self.Coef, z) # (batch_size, batch_size)*(batch_size, 1080)
			z_c_to_decoder = tf.reshape(z_c, z_from_encoder.shape) # (batch_size, 6, 6, 30)
			reconstruction = self.decoder(z_c_to_decoder)
			
			reconstruction_loss = 0.5*tf.reduce_sum(tf.pow(data - reconstruction, 2))
			
			s = tf.linalg.svd(self.Coef, compute_uv=False)
			s_ = 1-tf.exp(-s/self.gama)
			regularizer_loss = tf.reduce_sum(s_)
            
            # compute laplace maxtrix, then get laplace_loss
			absC = tf.abs(self.Coef)
			C = (absC + tf.transpose(absC)) * 0.5
			C = C + tf.eye(absC.shape[0])
			D = tf.linalg.diag(tf.sqrt((1.0 / tf.reduce_sum(C, axis=1))))
			I = tf.eye(D.shape[0])
			L = I - tf.matmul(tf.matmul(D, C), D)
			# x_flatten = tf.reshape(data, [data.shape[0], -1])
			# reconstruction_flatten = tf.reshape(reconstruction, [reconstruction.shape[0], -1])
			ZLZ = tf.matmul(tf.matmul(tf.transpose(z_c), L), z_c)
			laplace2_loss = 2.0 * tf.linalg.trace(ZLZ)
            

			selfexpression_loss = 0.5*tf.reduce_sum(tf.pow(z - z_c, 2))

			total_loss = self.lambda_reg*regularizer_loss + self.lambda_exp*selfexpression_loss + self.lambda_rec*reconstruction_loss + self.lambda_lap2*laplace2_loss

		grads = tape.gradient(total_loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		
		# update all kinds of loss 
		self.total_loss_tracker.update_state(total_loss)
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		self.selfexpression_loss_tracker.update_state(selfexpression_loss)
		self.regularizer_loss_tracker.update_state(regularizer_loss)
		self.laplace2_loss_tracker.update_state(laplace2_loss)
		return {
			"loss": self.total_loss_tracker.result(),
			"reconstruction_loss": self.reconstruction_loss_tracker.result(),
			"selfexpression_loss": self.selfexpression_loss_tracker.result(),
			"regularizer_loss": self.regularizer_loss_tracker.result(),
            "laplace2_loss": self.laplace2_loss_tracker.result()
		}

best_matrics = [0, 0, 0]
best_coef = np.ones(1)
def get_best_matrics(images, labels, lambdas, num_epoch, num_epoch_per_print, start_epoch, interval_epoch, init_C, datasetname):
	global best_matrics, best_coef
	print('----------------------------train: start----------------------------')
	begin = time.time()

	convAE = ConvAE(encoder=get_encoder(input_shape+[1], kernel_size, num_hidden), 
					decoder=get_decoder(input_shape_for_decoder, kernel_size, num_hidden), 	# # [6, 6, 30] [4, 4, 5]
					batch_size=num_images, lambdas=lambdas, gama=gama, init_C=init_C) 
	convAE.load_weights(pretrained_model)
	convAE.compile(optimizer=keras.optimizers.Adam())

	best_matrics = [0, 0, 0]
	best_coef = np.ones(1)
	# --------------------custom callback BEGIN--------------------
	class CustomCallback(keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs=None):
			global best_matrics, best_coef
			if epoch > start_epoch and epoch % interval_epoch  == 0:
				Coef = np.array(convAE.Coef)
				acc_nmi_purity = utils.cluster_and_getACC(Coef, labels, num_class=num_class, d=d, alpha=alpha, ro=ro)
				print("Epoch {}, reconstruction_loss: {}, regularizer_loss: {}, express_loss: {}, laplace2_loss: {}".format(epoch, logs['reconstruction_loss'], logs['regularizer_loss'], logs['selfexpression_loss'], logs['laplace2_loss']))
				print('Epoch: {} 	acc_nmi_purity***********************************************************: {}'.format(epoch, acc_nmi_purity))
				if acc_nmi_purity[0] > best_matrics[0]:
					best_coef = Coef
					best_matrics = acc_nmi_purity
			elif epoch % num_epoch_per_print == 0:
				print("Epoch {}, reconstruction_loss: {}, regularizer_loss: {}, express_loss: {}, laplace2_loss: {}".format(epoch, logs['reconstruction_loss'], logs['regularizer_loss'], logs['selfexpression_loss'], logs['laplace2_loss']))
	# --------------------custom callback END--------------------

	convAE.fit(images, epochs=num_epoch, batch_size=num_images,
			callbacks=[CustomCallback()], verbose=0, shuffle=False)

	# for epoch in range(num_epoch):
	# 	logs = convAE.train_step(images)
	# 	Coef = np.array(convAE.Coef)
	# 	acc_nmi_purity = utils.cluster_and_getACC(Coef, labels, num_class=num_class, d=d, alpha=alpha, ro=ro)
		
	# 	if epoch > start_epoch and epoch % interval_epoch  == 0:
	# 		Coef = np.array(convAE.Coef)
	# 		acc_nmi_purity = utils.cluster_and_getACC(Coef, labels, num_class=num_class, d=d, alpha=alpha, ro=ro)
	# 		print("Epoch {}, reconstruction_loss: {}, regularizer_loss: {}, express_loss: {}, laplace2_loss: {}".format(epoch, logs['reconstruction_loss'], logs['regularizer_loss'], logs['selfexpression_loss'], logs['laplace2_loss']))
	# 		print('Epoch: {} 	acc_nmi_purity***********************************************************: {}'.format(epoch, acc_nmi_purity))
	# 		if acc_nmi_purity[0] > best_matrics[0]:
	# 			best_matrics = acc_nmi_purity
	# 	elif epoch % num_epoch_per_print == 0:
	# 		print("Epoch {}, reconstruction_loss: {}, regularizer_loss: {}, express_loss: {}, laplace2_loss: {}".format(epoch, logs['reconstruction_loss'], logs['regularizer_loss'], logs['selfexpression_loss'], logs['laplace2_loss']))
	np.savetxt('Coef_{}_2.csv'.format(datasetname), best_coef) 
	print('Time used: %.1f s' % (time.time() - begin))
	print('----------------------------train: end----------------------------')
	return best_matrics


def test_orl():
	best_matrics = [0, 0, 0]
	best_parameters = []
	for lambda_rec in [1]:# 1 3 1: 85  1 8 0.5
		for lambda_reg in [8]: # [6, 7]
			for lambda_exp in [1]: # [2.5]
				for lambda_lap2 in [0.002, 0.001, 0.0005]: # 0.002
					lambdas = [lambda_rec, lambda_reg, lambda_exp, lambda_lap2]
					acc_nmi_purity = get_best_matrics(images, labels, lambdas,
								num_epoch=3500, num_epoch_per_print=5, start_epoch=2000, interval_epoch=1, init_C=1.0e-8, datasetname='orl')
					if acc_nmi_purity[0] > best_matrics[0]:
						best_matrics = acc_nmi_purity
						best_parameters = lambdas
	print('best_matrics: {} with best_parameters: {}'.format(best_matrics, best_parameters))
	with open('result_lap2', 'a') as file:
		file.write('orl: \nbest_matrics: {} with best_parameters: {}\n\n'.format(best_matrics, best_parameters))

def test_eyaleb():
	best_matrics = [0, 0, 0]
	best_parameters = []
	for lambda_rec in [1]:# 1 3 1: 85  1 8 0.5
		for lambda_reg in [1]: # [6, 7]
			for lambda_exp in [20]: # [2.5]
				for lambda_lap2 in [0.1, 0.001, 0.0001, 0.00001, 0.000001]: # 0.0001
					lambdas = [lambda_rec, lambda_reg, lambda_exp, lambda_lap2]
					acc_nmi_purity = get_best_matrics(images, labels, lambdas,
								num_epoch=5000, num_epoch_per_print=50, start_epoch=2500, interval_epoch=3, init_C=1.0e-8, datasetname='eyaleb')
					if acc_nmi_purity[0] > best_matrics[0]:
						best_matrics = acc_nmi_purity
						best_parameters = lambdas
	print('best_matrics: {} with best_parameters: {}'.format(best_matrics, best_parameters))
	with open('result_lap2', 'a') as file:
		file.write('yaleb: \nbest_matrics: {} with best_parameters: {}\n\n'.format(best_matrics, best_parameters))

def test_coil20():
	best_matrics = [0, 0, 0]
	best_parameters = []
	for lambda_rec in [0.2]:# 1 3 1: 85  1 8 0.5
		for lambda_reg in [20]: # [6, 7]
			for lambda_exp in [5]: # [2.5]
				for lambda_lap2 in [0.05, 0.001, 0.005]:# 0.001
					lambdas = [lambda_rec, lambda_reg, lambda_exp, lambda_lap2]
					acc_nmi_purity = get_best_matrics(images, labels, lambdas,
								num_epoch=100, num_epoch_per_print=1, start_epoch=15, interval_epoch=1, init_C=1.0e-8, datasetname='coil20')
					if acc_nmi_purity[0] > best_matrics[0]:
						best_matrics = acc_nmi_purity
						best_parameters = lambdas
	print('best_matrics: {} with best_parameters: {}'.format(best_matrics, best_parameters))
	with open('result_lap2', 'a') as file:
		file.write('coil20: \nbest_matrics: {} with best_parameters: {}\n\n'.format(best_matrics, best_parameters))

def test_umist():
	best_matrics = [0, 0, 0]
	best_parameters = []
	for lambda_rec in [2]:# 1 3 1: 85  1 8 0.5
		for lambda_reg in [1]: # [6, 7]
			for lambda_exp in [30]: # [2.5]
				for lambda_lap2 in [1.5, 1, 0.5]: # 1.5
					lambdas = [lambda_rec, lambda_reg, lambda_exp, lambda_lap2]
					acc_nmi_purity = get_best_matrics(images, labels, lambdas,
							num_epoch=700, num_epoch_per_print=1, start_epoch=10, interval_epoch=1, init_C=1.0e-8, datasetname='umist')
					if acc_nmi_purity[0] > best_matrics[0]:
						best_matrics = acc_nmi_purity
						best_parameters = lambdas
	print('best_matrics: {} with best_parameters: {}'.format(best_matrics, best_parameters))
	with open('result_lap2', 'a') as file:
		file.write('umist: \nbest_matrics: {} with best_parameters: {}\n\n'.format(best_matrics, best_parameters))

def test_mnist():
	best_matrics = [0, 0, 0]
	best_parameters = []	
	for lambda_rec in [10]:# 1 3 1: 85  1 8 0.5
		for lambda_reg in [8]: # [6, 7]
			for lambda_exp in [12]: # [2.5]
				for lambda_lap2 in [0.01, 0.002, 0.001]: # 0.01
					lambdas = [lambda_rec, lambda_reg, lambda_exp, lambda_lap2]
					acc_nmi_purity = get_best_matrics(images, labels, lambdas,
								num_epoch=140, num_epoch_per_print=1, start_epoch=50, interval_epoch=1, init_C=1.0e-8, datasetname='mnist')
					if acc_nmi_purity[0] > best_matrics[0]:
						best_matrics = acc_nmi_purity
						best_parameters = lambdas
	print('best_matrics: {} with best_parameters: {}'.format(best_matrics, best_parameters))
	with open('result_lap2', 'a') as file:
		file.write('mnist: \nbest_matrics: {} with best_parameters: {}\n\n'.format(best_matrics, best_parameters))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# datasets = ['UMIST_config.yaml', "ORL_config.yaml", 'MNIST_config.yaml', "COIL20_config.yaml", "EYaleB_config.yaml"]
datasets = ["EYaleB_config.yaml"]
for dataset in datasets:
	args = yaml.load(open(dataset, 'r'))

	num_class = args['dataset']['num_class']
	num_per_class =  args['dataset']['num_per_class']
	num_images = num_class * num_per_class
	data_path = args['dataset']['data_path']

	kernel_size = args['model']['kernel_size']
	num_hidden = args['model']['num_hidden']
	input_shape = args['model']['input_shape'] # [height, width]
	input_shape_for_decoder = args['model']['input_shape_for_decoder']
	pretrained_model = args['model']['pretrained_model']

	lambdas = args['training']['lambdas']
	gama = args['training']['gama']
	num_epoch = args['training']['num_epoch']
	num_epoch_per_print = args['training']['num_epoch_per_print']

	d = args['cluster']['d']
	alpha = args['cluster']['alpha']
	ro = args['cluster']['ro']


	# load data
	(images, labels) = utils.load_data(data_path, shape=input_shape) # ndarray
	images = tf.cast(images, dtype=tf.float32) # / 255

	images = tf.expand_dims(images, axis=-1) # below: 'images' is Tensor 
	if os.path.basename(data_path) == 'YaleBCrop025.mat':
		input_shape = [48, 48]
		images = tf.image.resize(images, input_shape) # resize to this specific size to fit Conv and Deconv

	if os.path.basename(data_path) == 'mnist1000.mat':
		input_shape = [32, 32]
		images = tf.image.resize(images, input_shape)

	if os.path.basename(data_path) == 'YaleBCrop025.mat':
		test_eyaleb()
	elif os.path.basename(data_path) == 'ORL_32x32.mat':
		test_orl()
	elif os.path.basename(data_path) == 'COIL20.mat':
		test_coil20()
	elif os.path.basename(data_path) == 'umist-32-32.mat':
		test_umist()
	elif os.path.basename(data_path) == 'mnist1000.mat':
		test_mnist()
	# encoder=get_encoder(input_shape+[1], kernel_size, num_hidden)
	# decoder=get_decoder([4, 4, 5], kernel_size, num_hidden)
	# encoder.summary()
	# decoder.summary()