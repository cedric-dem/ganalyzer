from config import *

from keras import layers
import tensorflow as tf
import math
from ganalyzer.model_config import *

def _make_config(*, gen_base, gen_min, disc_seq, disc_fc, gen_ch0, gen_pre_dense = None, extra_conv = False):
	return {
		"gen_base": gen_base,
		"gen_min": gen_min,
		"disc_seq": disc_seq,
		"disc_fc": disc_fc,
		"gen_ch0": gen_ch0,
		"gen_pre_dense": gen_pre_dense or [],
		"extra_conv": extra_conv,
	}

def _clone_configs(configs):
	return {name: dict(cfg) for name, cfg in configs.items()}

MODEL_CONFIGS_64 = {
	"model_0_tiny": _make_config(
		gen_base = [256, 128, 64, 32, 16, 8],
		gen_min = 8,
		disc_seq = [32, 64, 128, 256, 256, 256],
		disc_fc = lambda image_size: [],
		gen_ch0 = 256,
	),
	"model_0_with_add": _make_config(
		gen_base = [256, 128, 64, 32, 16, 8],
		gen_min = 8,
		disc_seq = [32, 64, 128, 256, 256, 256],
		disc_fc = lambda image_size: [additional_dense_units] * 3,
		gen_ch0 = 256,
		gen_pre_dense = [additional_dense_units] * 3,
	),
	"model_1_small": _make_config(
		gen_base = [336, 168, 84, 42, 21, 10],
		gen_min = 10,
		disc_seq = [42, 84, 168, 336, 336, 336],
		disc_fc = lambda image_size: [image_size * 2],
		gen_ch0 = 336,
		gen_pre_dense = [320],
		extra_conv = True,
	),
	"model_2_medium": _make_config(
		gen_base = [416, 208, 104, 52, 26, 13],
		gen_min = 13,
		disc_seq = [52, 104, 208, 416, 416, 416],
		disc_fc = lambda image_size: [image_size * 3],
		gen_ch0 = 416,
		gen_pre_dense = [640],
		extra_conv = True,
	),
}

MODEL_CONFIGS_80 = {
	"model_0_tiny": _make_config(
		gen_base = [288, 144, 72, 36, 18, 9],
		gen_min = 9,
		disc_seq = [36, 72, 144, 288, 288, 288],
		disc_fc = lambda image_size: [],
		gen_ch0 = 288,
	),
	"model_0_with_add": _make_config(
		gen_base = [288, 144, 72, 36, 18, 9],
		gen_min = 9,
		disc_seq = [36, 72, 144, 288, 288, 288],
		disc_fc = lambda image_size: [additional_dense_units] * 3,
		gen_ch0 = 288,
		gen_pre_dense = [additional_dense_units] * 3,
	),
	"model_1_small": _make_config(
		gen_base = [384, 192, 96, 48, 24, 12],
		gen_min = 12,
		disc_seq = [48, 96, 192, 384, 384, 384],
		disc_fc = lambda image_size: [image_size * 2],
		gen_ch0 = 384,
		gen_pre_dense = [352],
		extra_conv = True,
	),
	"model_2_medium": _make_config(
		gen_base = [480, 240, 120, 60, 30, 15],
		gen_min = 15,
		disc_seq = [60, 120, 240, 480, 480, 480],
		disc_fc = lambda image_size: [image_size * 3],
		gen_ch0 = 480,
		gen_pre_dense = [704],
		extra_conv = True,
	),
}

MODEL_CONFIGS_100 = {
	"model_0_tiny": _make_config(
		gen_base = [320, 160, 80, 40, 20, 10],
		gen_min = 10,
		disc_seq = [40, 80, 160, 320, 320, 320],
		disc_fc = lambda image_size: [],
		gen_ch0 = 320,
	),
	"model_0_with_add": _make_config(
		gen_base = [320, 160, 80, 40, 20, 10],
		gen_min = 10,
		disc_seq = [40, 80, 160, 320, 320, 320],
		disc_fc = lambda image_size: [additional_dense_units] * 3,
		gen_ch0 = 320,
		gen_pre_dense = [additional_dense_units] * 3,
	),
	"model_1_small": _make_config(
		gen_base = [424, 212, 106, 53, 26, 14],
		gen_min = 13,
		disc_seq = [53, 106, 212, 424, 424, 424],
		disc_fc = lambda image_size: [image_size * 2],
		gen_ch0 = 424,
		gen_pre_dense = [448],
		extra_conv = True,
	),
	"model_2_medium": _make_config(
		gen_base = [528, 264, 132, 66, 33, 17],
		gen_min = 17,
		disc_seq = [66, 132, 264, 528, 528, 528],
		disc_fc = lambda image_size: [image_size * 4],
		gen_ch0 = 528,
		gen_pre_dense = [896],
		extra_conv = True,
	),
}

MODEL_CONFIGS_114 = {
	"model_0_tiny": _make_config(
		gen_base = [352, 176, 88, 44, 22, 11, 11],
		gen_min = 11,
		disc_seq = [44, 88, 176, 352, 352, 352, 352],
		disc_fc = lambda image_size: [],
		gen_ch0 = 352,
	),
	"model_0_with_add": _make_config(
		gen_base = [352, 176, 88, 44, 22, 11, 11],
		gen_min = 11,
		disc_seq = [44, 88, 176, 352, 352, 352, 352],
		disc_fc = lambda image_size: [additional_dense_units] * 3,
		gen_ch0 = 352,
		gen_pre_dense = [additional_dense_units] * 3,
	),
	"model_1_small": _make_config(
		gen_base = [472, 236, 118, 59, 30, 15, 15],
		gen_min = 15,
		disc_seq = [59, 118, 236, 472, 472, 472, 472],
		disc_fc = lambda image_size: [96],
		gen_ch0 = 472,
		gen_pre_dense = [320],
		extra_conv = True,
	),
	"model_2_medium": _make_config(
		gen_base = [592, 296, 148, 74, 37, 19, 19],
		gen_min = 19,
		disc_seq = [74, 148, 296, 592, 592, 592, 592],
		disc_fc = lambda image_size: [192],
		gen_ch0 = 592,
		gen_pre_dense = [640],
		extra_conv = True,
	),
}

MODEL_CONFIGS_120 = {
	"model_0_tiny": _make_config(
		gen_base = [384, 192, 96, 48, 24, 12, 12],
		gen_min = 12,
		disc_seq = [48, 96, 192, 384, 384, 384, 384],
		disc_fc = lambda image_size: [],
		gen_ch0 = 384,
	),
	"model_0_with_add": _make_config(
		gen_base = [384, 192, 96, 48, 24, 12, 12],
		gen_min = 12,
		disc_seq = [48, 96, 192, 384, 384, 384, 384],
		disc_fc = lambda image_size: [additional_dense_units] * 3,
		gen_ch0 = 384,
		gen_pre_dense = [additional_dense_units] * 3,
	),
	"model_1_small": _make_config(
		gen_base = [512, 256, 128, 64, 32, 16, 16],
		gen_min = 16,
		disc_seq = [64, 128, 256, 512, 512, 512, 512],
		disc_fc = lambda image_size: [96],
		gen_ch0 = 512,
		gen_pre_dense = [384],
		extra_conv = True,
	),
	"model_2_medium": _make_config(
		gen_base = [640, 320, 160, 80, 40, 20, 20],
		gen_min = 20,
		disc_seq = [80, 160, 320, 640, 640, 640, 640],
		disc_fc = lambda image_size: [192],
		gen_ch0 = 640,
		gen_pre_dense = [768],
		extra_conv = True,
	),
}

MODEL_CONFIGS_128 = {
	"model_0_tiny": _make_config(
		gen_base = [416, 208, 104, 52, 26, 13, 13],
		gen_min = 13,
		disc_seq = [52, 104, 208, 416, 416, 416, 416],
		disc_fc = lambda image_size: [],
		gen_ch0 = 416,
	),
	"model_0_with_add": _make_config(
		gen_base = [416, 208, 104, 52, 26, 13, 13],
		gen_min = 13,
		disc_seq = [52, 104, 208, 416, 416, 416, 416],
		disc_fc = lambda image_size: [additional_dense_units] * 3,
		gen_ch0 = 416,
		gen_pre_dense = [additional_dense_units] * 3,
	),
	"model_1_small": _make_config(
		gen_base = [560, 280, 140, 70, 35, 18, 18],
		gen_min = 18,
		disc_seq = [70, 140, 280, 560, 560, 560, 560],
		disc_fc = lambda image_size: [112],
		gen_ch0 = 560,
		gen_pre_dense = [448],
		extra_conv = True,
	),
	"model_2_medium": _make_config(
		gen_base = [704, 352, 176, 88, 44, 22, 22],
		gen_min = 22,
		disc_seq = [88, 176, 352, 704, 704, 704, 704],
		disc_fc = lambda image_size: [224],
		gen_ch0 = 704,
		gen_pre_dense = [896],
		extra_conv = True,
	),
}

MODEL_CONFIGS_240 = {
	"model_0_tiny": _make_config(
		gen_base = [256, 192, 128, 96, 64, 32],
		gen_min = 32,
		disc_seq = [64, 96, 128, 192, 256, 256],
		disc_fc = lambda image_size: [image_size * 2],
		gen_ch0 = 512,
		extra_conv = True,
	),
	"model_0_with_add": _make_config(
		gen_base = [256, 192, 128, 96, 64, 32],
		gen_min = 32,
		disc_seq = [64, 96, 128, 192, 256, 256],
		disc_fc = lambda image_size: [additional_dense_units] * 3,
		gen_ch0 = 512,
		gen_pre_dense = [additional_dense_units] * 3,
		extra_conv = True,
	),
	"model_1_small": _make_config(
		gen_base = [408, 292, 204, 144, 104, 60],
		gen_min = 60,
		disc_seq = [84, 144, 204, 292, 408, 408],
		disc_fc = lambda image_size: [image_size * 3],
		gen_ch0 = 616,
		gen_pre_dense = [800],
		extra_conv = True,
	),
	"model_2_medium": _make_config(
		gen_base = [560, 392, 280, 192, 144, 88],
		gen_min = 88,
		disc_seq = [104, 192, 280, 392, 560, 560],
		disc_fc = lambda image_size: [image_size * 4],
		gen_ch0 = 720,
		gen_pre_dense = [1600],
		extra_conv = True,
	),
}

MODEL_CONFIGS_BY_SIZE = {
	64: MODEL_CONFIGS_64,
	80: MODEL_CONFIGS_80,
	100: MODEL_CONFIGS_100,
	114: MODEL_CONFIGS_114,
	120: MODEL_CONFIGS_120,
	128: MODEL_CONFIGS_128,
	240: MODEL_CONFIGS_240,
}

MODEL_CONFIGS = MODEL_CONFIGS_BY_SIZE[model_output_size]

def _num_upsamples_to_reach(img_size, base = 4):
	if img_size < base:
		raise ValueError("img_size should be >= 4")
	cur = base
	ups = 0
	while cur < img_size:
		cur *= 2
		ups += 1
	return ups, cur

def _filters_for_gen(step_idx, base, min_filter):
	return base[step_idx] if step_idx < len(base) else max(min_filter, base[-1] // 2)

def _filters_for_disc(step_idx, seq):
	return seq[step_idx] if step_idx < len(seq) else seq[-1]

def _downsampling_steps(image_size):
	steps = 0
	cur = image_size
	while cur > 4:
		cur = math.ceil(cur / 2)
		steps += 1
	return steps

def _disc_feature_width(seq, image_size):
	return _filters_for_disc(_downsampling_steps(image_size) - 1, seq)

def _auto_disc_sequence(cfg, image_size):
	return [_filters_for_gen(i, cfg["gen_base"], cfg["gen_min"]) for i in range(_downsampling_steps(image_size))]

def _scale_filters(seq, scale):
	return [max(4, int(round(val * scale))) for val in seq]

def _build_discriminator(image_size, disc_seq, disc_fc, *, extra_fc_units = 0):
	inputs = layers.Input(shape = (image_size, image_size, 3))
	x = inputs

	cur = image_size
	step = 0
	while cur > 4:
		x = layers.Conv2D(_filters_for_disc(step, disc_seq), kernel_size = 5, strides = 2, padding = "same")(x)
		x = layers.LeakyReLU(alpha = 0.2)(x)
		x = layers.Dropout(0.3)(x)
		cur = math.ceil(cur / 2)
		step += 1

	x = layers.GlobalAveragePooling2D()(x)
	for size in disc_fc(image_size):
		x = layers.Dense(size)(x)
		x = layers.LeakyReLU(alpha = 0.2)(x)
		x = layers.Dropout(0.3)(x)

	if extra_fc_units > 0:
		x = layers.Dense(extra_fc_units)(x)
		x = layers.LeakyReLU(alpha = 0.2)(x)
		x = layers.Dropout(0.3)(x)

	outputs = layers.Dense(1, activation = "sigmoid")(x)

	return tf.keras.Model(inputs, outputs, name = f"Discriminator_{image_size}")

def get_discriminator():
	cfg = MODEL_CONFIGS[model_name]
	image_size = int(dataset_dimension)

	assert image_size == int(image_size)
	disc_seq = cfg["disc_seq"] or _auto_disc_sequence(cfg, image_size)

	gen_params = get_generator().count_params()
	base_discriminator = _build_discriminator(image_size, disc_seq, cfg["disc_fc"])
	base_disc_params = base_discriminator.count_params()

	if base_disc_params == gen_params:
		return base_discriminator

	scale = math.sqrt(gen_params / base_disc_params)
	scaled_seq = _scale_filters(disc_seq, scale)
	discriminator = _build_discriminator(image_size, scaled_seq, cfg["disc_fc"])
	disc_params = discriminator.count_params()

	if disc_params != gen_params:
		missing = gen_params - disc_params
		if missing > 0:
			feature_width = _disc_feature_width(scaled_seq, image_size)
			extra_units = missing // (feature_width + 1)
			if extra_units > 0:
				discriminator = _build_discriminator(image_size, scaled_seq, cfg["disc_fc"], extra_fc_units = extra_units)
				disc_params = discriminator.count_params()

	return discriminator

def get_generator():
	cfg = MODEL_CONFIGS[model_name]
	image_size = int(dataset_dimension)
	latent_dim = latent_dimension_generator
	base_spatial = 4

	assert image_size == int(image_size)
	ups, reached = _num_upsamples_to_reach(image_size, base = base_spatial)

	inputs = layers.Input(shape = (latent_dim,))
	x = inputs

	for size in cfg.get("gen_pre_dense", []):
		x = layers.Dense(size, use_bias = False)(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)

	ch0 = cfg["gen_ch0"]
	x = layers.Dense(base_spatial * base_spatial * ch0, use_bias = False)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)
	x = layers.Reshape((base_spatial, base_spatial, ch0))(x)

	for i in range(ups):
		x = layers.Conv2DTranspose(_filters_for_gen(i, cfg["gen_base"], cfg["gen_min"]), kernel_size = 4, strides = 2, padding = "same", use_bias = False)(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)

	if reached != image_size:
		x = layers.Conv2D(_filters_for_gen(ups, cfg["gen_base"], cfg["gen_min"]), kernel_size = 3, padding = "same", use_bias = False)(x)
		x = layers.LeakyReLU()(x)
		x = layers.Resizing(image_size, image_size, interpolation = "bilinear")(x)

	if cfg.get("extra_conv", False):
		x = layers.Conv2D(32, kernel_size = 3, padding = "same", use_bias = False)(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)

	outputs = layers.Conv2D(3, kernel_size = 3, strides = 1, padding = "same", activation = "tanh")(x)
	return tf.keras.Model(inputs, outputs, name = f"Generator_{image_size}")
