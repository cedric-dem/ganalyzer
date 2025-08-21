from config import *

from keras import layers
import tensorflow as tf
import math

if model_name == "model_a":
	def _num_upsamples_to_reach(img_size, base = 4):
		if img_size < base:
			raise ValueError("img_size should be >= 4")
		cur = base
		ups = 0
		while cur < img_size:
			cur *= 2
			ups += 1
		return ups, cur

	def _filters_for_gen(step_idx, total_steps):
		base = [512, 256, 128, 64, 32, 16]
		return base[step_idx] if step_idx < len(base) else max(16, base[-1] // 2)

	def _filters_for_disc(step_idx):
		seq = [64, 128, 256, 512, 512, 512]
		return seq[step_idx] if step_idx < len(seq) else 512

	def get_discriminator():
		image_size = int(dataset_dimension)
		fc_size = 128

		assert image_size == int(image_size)
		inputs = layers.Input(shape = (image_size, image_size, 3))
		x = inputs

		cur = image_size
		step = 0
		while cur > 4:
			x = layers.Conv2D(_filters_for_disc(step), kernel_size = 5, strides = 2, padding = "same")(x)
			x = layers.LeakyReLU(alpha = 0.2)(x)
			x = layers.Dropout(0.3)(x)
			cur = math.ceil(cur / 2)
			step += 1

		x = layers.GlobalAveragePooling2D()(x)
		x = layers.Dense(fc_size)(x)
		x = layers.LeakyReLU(alpha = 0.2)(x)
		x = layers.Dropout(0.3)(x)
		outputs = layers.Dense(1, activation = "sigmoid")(x)

		return tf.keras.Model(inputs, outputs, name = f"Discriminator_{image_size}")

	def get_generator():
		image_size = int(dataset_dimension)
		latent_dim = latent_dimension_generator
		base_spatial = 4

		assert image_size == int(image_size)
		ups, reached = _num_upsamples_to_reach(image_size, base = base_spatial)

		inputs = layers.Input(shape = (latent_dim,))
		x = inputs

		ch0 = 512
		x = layers.Dense(base_spatial * base_spatial * ch0, use_bias = False)(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)
		x = layers.Reshape((base_spatial, base_spatial, ch0))(x)

		for i in range(ups):
			x = layers.Conv2DTranspose(
				_filters_for_gen(i, ups),
				kernel_size = 4, strides = 2, padding = "same", use_bias = False
			)(x)
			x = layers.BatchNormalization()(x)
			x = layers.LeakyReLU()(x)

		if reached != image_size:
			x = layers.Conv2D(_filters_for_gen(ups, ups + 1), kernel_size = 3, padding = "same", use_bias = False)(x)
			x = layers.LeakyReLU()(x)
			x = layers.Resizing(image_size, image_size, interpolation = "bilinear")(x)

		x = layers.Conv2D(32, kernel_size = 3, padding = "same", use_bias = False)(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)

		outputs = layers.Conv2D(3, kernel_size = 3, strides = 1, padding = "same", activation = "tanh")(x)
		return tf.keras.Model(inputs, outputs, name = f"Generator_{image_size}")

else:
	raise Exception("model not found")
