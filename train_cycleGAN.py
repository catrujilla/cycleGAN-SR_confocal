# training a cycleGAN on Ana Doblas' experimental confocal dataset RED
'''
This Python script implements a CycleGAN (Cycle-Consistent Generative Adversarial Network) to perform image translation between two domains of The Memphis Unviersity experimental confocal microscopy dataset. 
The CycleGAN architecture involves two generators and two discriminators, enabling the transformation of images from one domain (e.g., raw confocal images) to another (e.g., enhanced or processed versions).

Key Features:
Model Architecture: Defines a discriminator and generator using convolutional neural networks (CNNs), incorporating instance normalization and residual blocks (resnet blocks) for effective image translation.

Custom Loss Functions: Includes adversarial loss to train the generators to produce realistic images and perceptual loss based on VGG19 feature maps to ensure perceptual similarity between generated and target images.

Training Process: Alternates between updating discriminators and generators based on adversarial and cycle-consistency losses. It includes functions for image pooling, model saving, and performance monitoring through metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

Dataset Handling: Loads and preprocesses training and validation datasets (confocal_exper_altogether_trainR_256.npz, confocal_exper_non_sat_filt_validR_256.npz, confocal_exper_paired_filt_validsetR_256.npz), ensuring compatibility with TensorFlow/Keras requirements.

Utilities: Provides functions for visualizing generated images during training (summarize_performance), updating image pools (update_image_pool), and saving models (save_models) at specified intervals.

Usage:
Ensure all necessary Python libraries (tensorflow, numpy, matplotlib, tensorflow_addons) are installed.
Customize dataset paths and filenames according to the specific dataset location and setup.
Execute the script in a suitable Python environment capable of GPU acceleration for faster training.

Authors: Dr. Ana Doblas and Dr. Carlos Trujillo

Affiliations: University of Massachussets Dartmouth, Universidad EAFIT

'''

import sys

print("Python version")
print(sys.version)

# training a cycleGAN on Ana Doblas' experimental confocal dataset
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint

from matplotlib import pyplot
from metrics import avg_SSIM
from metrics import bgstd_batch
from metrics import psnr
from metrics import cutoff_batch

import tensorflow.keras.backend as K

#New imports
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Activation, Concatenate
from tensorflow_addons.layers import InstanceNormalization

import tensorflow as tf

import keras

print(tf.__version__)
print(keras.__version__)

# Load pre-trained VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, input_shape=(256, 256, 3))

# Freeze all layers in VGG19
for layer in vgg.layers:
    layer.trainable = False
    
gpus = tf.config.list_logical_devices('GPU')
#strategy = tf.distribute.MirroredStrategy(gpus)

print("Number of GPUs Available: ", len(gpus))
for gpu in gpus:
    print("GPU Name:", gpu.name)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

# define the standalone generator model
def define_generator(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model


'''
Customizing the adversarial loss function to incorporate the spatial frequency content of the generated image is possible and can potentially improve the quality of the generated images.

One way to incorporate spatial frequency content into the adversarial loss function is to use a perceptual loss function. Perceptual loss functions measure the perceptual similarity between the generated and target images by comparing their feature representations extracted from a pre-trained deep neural network.

We can modify the adversarial loss in the define_composite_model function by defining a new loss function that combines the mean squared error and the perceptual loss. 
'''

import tensorflow.image as tf_image
'''
def perceptual_loss(y_true, y_pred):
    # Compute the structural similarity index between the generated and target images
    ssim_loss = 1.0 - tf_image.ssim(y_true, y_pred, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    # Return the perceptual loss as a weighted sum of the SSIM loss and the adversarial loss
    return ssim_loss + 0.01 * adversarial_loss(y_true, y_pred)
    #return ssim_loss
'''

def perceptual_loss(y_true, y_pred):
	# Get output feature maps from VGG19 for the generated and target images
	fake_feat = vgg(y_pred)
	real_feat = vgg(y_true)
	# Compute the mean squared error between the feature maps
	mse_loss = K.mean(K.square(fake_feat - real_feat))
	# Return the perceptual loss as a weighted sum of the mean squared error and the adversarial loss
	return mse_loss + 0.01 * adversarial_loss(y_true, y_pred)


def adversarial_loss(y_true, y_pred):
	# Compute the mean squared error between the discriminator's prediction and a vector of ones
	return K.mean(K.square(y_pred - K.ones_like(y_pred)))

'''
In the model.compile() method call, the second 'mae' loss function in the loss list corresponds to the identity loss term in the cycle GAN, which measures the difference between the original input image and the reconstructed image after passing through both generators.

More specifically, during training, the composite model minimizes four loss functions:

Adversarial loss (mse): This loss is used to train the generator to produce images that can fool the discriminator into classifying them as real. The generator aims to minimize this loss.
Identity loss (mae): This loss is used to encourage the generator to preserve the input image's identity during the image translation process. The generator tries to minimize this loss by ensuring that the output of the first generator is as close as possible to the input image.
Forward cycle consistency loss (mae): This loss is used to ensure that the output image of the first generator can be translated back to the original input image by the second generator. The first generator tries to minimize this loss.
Backward cycle consistency loss (mae): This loss is used to ensure that the output image of the second generator can be translated back to the original input image by the first generator. The second generator tries to minimize this loss.
The identity loss term is used to ensure that the generator does not change the input image's essential characteristics, such as the content, style, or structure, while generating the output image. By minimizing the identity loss, the generator can learn to maintain the essential characteristics of the input image in the generated output image, even when the input image and output image have different styles, colors, or other features.
'''

'''
In the CycleGAN architecture, the generator that is being trained to generate images that can fool the discriminator is the first generator (g_model_1), which is the generator that maps images from domain A to domain B. The second generator (g_model_2), which maps images from domain B to domain A, does not contribute to the adversarial loss.

The adversarial loss is a part of the training process for the generator, which aims to generate images that can deceive the discriminator into classifying them as real, while the discriminator tries to distinguish between the generated and real images accurately. By minimizing the adversarial loss, the generator learns to produce images that look like they belong to the target domain and can be used for image translation.

During the adversarial training process, the generator minimizes the adversarial loss by updating its weights and biases. In contrast, the discriminator minimizes the adversarial loss by updating its own weights and biases to better distinguish between the generated and real images.
'''

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# Model creation
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity element
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	#model.compile(loss=[custom_loss_function, 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	model.compile(loss=[adversarial_loss, perceptual_loss, perceptual_loss, perceptual_loss], loss_weights=[1, 1, 10, 5], optimizer=opt)
	#model.compile(loss=[adversarial_loss, perceptual_loss, perceptual_loss, perceptual_loss], loss_weights=[1, 5, 5, 5], optimizer=opt)
	return model


# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	#g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0

	# plot real images
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])

	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)
    

# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, trainsubset, validset, validset_paired):
	# define properties of the training run
	n_epochs, n_batch, = 80, 1
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# prepare image pool for fakes
	poolA, poolB = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs

	number_images = dataset[0].shape[0]
	number_imagesV = validset[0].shape[0]
	number_imagesV_paired = validset_paired[0].shape[0]
    
	with open('metrics_values.txt', 'w') as f:

		for i in range(n_steps):
			# select a batch of real samples
			X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
			X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
			# generate a batch of fake samples
			X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
			X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
			# update fakes from pool
			X_fakeA = update_image_pool(poolA, X_fakeA)
			X_fakeB = update_image_pool(poolB, X_fakeB)
			# update generator B->A via adversarial and cycle loss
			g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
			# update discriminator for A -> [real/fake]
			dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
			dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
			# update generator A->B via adversarial and cycle loss
			g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
			# update discriminator for B -> [real/fake]
			dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
			dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
			# summarize performance
			print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
			# evaluate the model performance every so often
			if (i+1) % (bat_per_epo * 1) == 0:
				# plot A->B translation
				summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
				# plot B->A translation
				summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
			if (i+1) % (bat_per_epo * 1) == 0:
				# save the models
				save_models(i, g_model_AtoB, g_model_BtoA)
			if (i + 1) % (bat_per_epo) == 0:

				cutoff_list, cutoff_std_list = cutoff_batch(g_model_AtoB, trainsubset,image_shape,n_samples=number_images)
				
				std_list, std_std_list = bgstd_batch(g_model_AtoB, trainsubset, trainsubset, n_samples=number_images)
				psnr_listv, psnr_listv_std, mse_listv, mse_listv_std = psnr(g_model_AtoB,validset_paired,image_shape,n_samples=number_imagesV_paired)
				cutoff_listv, cutoff_std_listv = cutoff_batch(g_model_AtoB,validset,image_shape,n_samples=number_imagesV)
				avg_ssimv, avg_ssimv_std = avg_SSIM(g_model_AtoB, validset_paired, image_shape, n_samples=number_imagesV_paired)
				
				std_listv, std_std_listv = bgstd_batch(g_model_AtoB, validset, validset, n_samples=number_imagesV)

				f.write(str(cutoff_list) + ' ' + str(cutoff_std_list) + ' ' + str(cutoff_listv) + ' ' + str(cutoff_std_listv) + ' ' + str(std_list) + ' ' + str(std_std_list) + ' ' + str(std_listv) + ' ' + str(std_std_listv) + ' ' + str(psnr_listv) + ' ' + str(psnr_listv_std) + ' ' + str(mse_listv) + ' ' + str(mse_listv_std) + ' ' + str(avg_ssimv) + ' ' + str(avg_ssimv_std) +  '\n')
				# flush the data to the file
				f.flush()

# load image data
dataset = load_real_samples('confocal_exper_altogether_trainR_256.npz')
#trainsubset = load_real_samples('confocal_exper_medium_trainsubset_512.npz')
validset = load_real_samples('confocal_exper_non_sat_filt_validR_256.npz')
validset_paired = load_real_samples('confocal_exper_paired_filt_validsetR_256.npz')

print('Loaded', dataset[0].shape, dataset[1].shape)

print('Number of images for training:', dataset[0].shape[0])

# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, dataset, validset, validset_paired)