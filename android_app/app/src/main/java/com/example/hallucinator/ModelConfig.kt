package com.example.hallucinator

object ModelConfig {
    const val DECODER_INPUT_SIZE = 6272

    const val MODEL_IMAGE_SIZE = 224

    const val DECODER_IMAGE_WIDTH = 224
    const val DECODER_IMAGE_HEIGHT = 224

    const val DECODER_IMAGE_CHANNELS = 3

    const val DECODER_OUTPUT_SIZE = DECODER_IMAGE_WIDTH * DECODER_IMAGE_HEIGHT * DECODER_IMAGE_CHANNELS
}