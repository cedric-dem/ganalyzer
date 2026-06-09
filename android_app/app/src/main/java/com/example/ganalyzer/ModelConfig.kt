package com.example.ganalyzer

object ModelConfig {
    const val DISCRIMINATOR_PATH = "discriminator.tflite"
    const val GENERATOR_PATH = "generator.tflite"

    const val IMAGES_SIZE = 100
    const val DECODER_IMAGE_CHANNELS = 3

    const val PREVIEW_GRID_SIZE = 10
    const val LATENT_SPACE_SIZE = PREVIEW_GRID_SIZE * PREVIEW_GRID_SIZE


}