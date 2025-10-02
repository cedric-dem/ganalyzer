package com.example

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileNotFoundException
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.roundToInt
import kotlin.random.Random

class GeneratorApplicator(context: Context) : Closeable {

    private val interpreter: Interpreter?
    private val latentVectorLength: Int
    private val outputElementCount: Int
    private val outputDimensions: OutputDimensions?
    private val inputShape: IntArray
    private val outputShape: IntArray
    private val supportsSingleBatchInput: Boolean
    private val supportsSingleBatchOutput: Boolean

    init {
        val modelBuffer = loadModelFile(context.assets, MODEL_FILE_NAME)

        if (modelBuffer != null) {
            interpreter = Interpreter(modelBuffer)

            inputShape = interpreter.getInputTensor(0).shape()
            val inputBatch = if (inputShape.size > 1) inputShape[0] else 1
            latentVectorLength = if (inputShape.isEmpty()) {
                0
            } else if (inputShape.size == 1) {
                inputShape[0]
            } else {
                inputShape.drop(1).fold(1) { acc, dim -> acc * dim }
            }
            supportsSingleBatchInput = inputBatch == 1 && latentVectorLength > 0

            outputShape = interpreter.getOutputTensor(0).shape()
            val outputBatch = if (outputShape.size > 1) outputShape[0] else 1
            outputElementCount = if (outputShape.isEmpty()) {
                0
            } else if (outputShape.size == 1) {
                outputShape[0]
            } else {
                outputShape.drop(1).fold(1) { acc, dim -> acc * dim }
            }
            supportsSingleBatchOutput = outputBatch == 1 && outputElementCount > 0
            outputDimensions = inferOutputDimensions(outputShape)
        } else {
            interpreter = null

            latentVectorLength = FALLBACK_LATENT_VECTOR_LENGTH
            outputDimensions = OutputDimensions(
                width = FALLBACK_IMAGE_WIDTH,
                height = FALLBACK_IMAGE_HEIGHT,
                channels = FALLBACK_IMAGE_CHANNELS,
                channelsLast = true,
            )
            outputElementCount = outputDimensions.width * outputDimensions.height * outputDimensions.channels

            inputShape = intArrayOf(latentVectorLength)
            outputShape = intArrayOf(
                outputDimensions.height,
                outputDimensions.width,
                outputDimensions.channels,
            )
            supportsSingleBatchInput = true
            supportsSingleBatchOutput = true
        }
    }

    fun expectedInputSize(): Int = latentVectorLength

    fun apply(input: FloatArray): FloatArray? {
        if (!supportsSingleBatchInput || !supportsSingleBatchOutput) {
            Log.w(TAG, "Model does not support single-batch execution. Input shape=${inputShape.contentToString()} output shape=${outputShape.contentToString()}")
            return null
        }

        if (input.size != latentVectorLength) {
            Log.w(TAG, "Input vector length ${input.size} does not match expected $latentVectorLength")
            return null
        }

        val activeInterpreter = interpreter
        if (activeInterpreter == null) {
            return generateFallbackOutput(input)
        }

        val inputBuffer = ByteBuffer.allocateDirect(latentVectorLength * BYTES_PER_FLOAT).order(ByteOrder.nativeOrder())
        input.forEach { value ->
            inputBuffer.putFloat(value)
        }
        inputBuffer.rewind()

        val outputBuffer = ByteBuffer.allocateDirect(outputElementCount * BYTES_PER_FLOAT).order(ByteOrder.nativeOrder())

        return try {
            synchronized(activeInterpreter) {
                activeInterpreter.run(inputBuffer, outputBuffer)
            }
            outputBuffer.rewind()
            val floatBuffer: FloatBuffer = outputBuffer.asFloatBuffer()
            val result = FloatArray(floatBuffer.remaining())
            floatBuffer.get(result)
            result
        } catch (throwable: Throwable) {
            Log.e(TAG, "Error while running generator model", throwable)
            null
        }
    }

    fun applyToBitmap(input: FloatArray): Bitmap? {
        val rawOutput = apply(input) ?: return null
        val dimensions = outputDimensions ?: run {
            Log.w(TAG, "Unable to determine output dimensions for shape ${outputShape.contentToString()}")
            return null
        }

        val requiredValues = dimensions.width * dimensions.height * dimensions.channels
        if (rawOutput.size < requiredValues) {
            Log.w(TAG, "Model returned ${rawOutput.size} values but $requiredValues are required to create an image")
            return null
        }

        val normalization = determineNormalization(rawOutput)
        val bitmap = Bitmap.createBitmap(dimensions.width, dimensions.height, Bitmap.Config.ARGB_8888)

        if (dimensions.channelsLast) {
            fillBitmapChannelsLast(bitmap, rawOutput, dimensions, normalization)
        } else {
            fillBitmapChannelsFirst(bitmap, rawOutput, dimensions, normalization)
        }

        return bitmap
    }

    override fun close() {
        interpreter?.close()
    }

    private fun fillBitmapChannelsLast(
        bitmap: Bitmap,
        values: FloatArray,
        dimensions: OutputDimensions,
        normalization: NormalizationConfig,
    ) {
        val width = dimensions.width
        val height = dimensions.height
        val channels = dimensions.channels

        for (y in 0 until height) {
            for (x in 0 until width) {
                val baseIndex = (y * width + x) * channels
                val firstChannel = values[baseIndex]
                val secondChannel = if (channels > 1) values[baseIndex + 1] else firstChannel
                val thirdChannel = if (channels > 2) values[baseIndex + 2] else firstChannel

                val r = toColorComponent(firstChannel, normalization)
                val g = toColorComponent(secondChannel, normalization)
                val b = toColorComponent(thirdChannel, normalization)
                bitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }
    }

    private fun fillBitmapChannelsFirst(
        bitmap: Bitmap,
        values: FloatArray,
        dimensions: OutputDimensions,
        normalization: NormalizationConfig,
    ) {
        val width = dimensions.width
        val height = dimensions.height
        val channels = dimensions.channels
        val planeSize = width * height

        for (y in 0 until height) {
            for (x in 0 until width) {
                val offset = y * width + x
                val firstChannel = values[offset]
                val secondChannel = if (channels > 1) values[planeSize + offset] else firstChannel
                val thirdChannel = if (channels > 2) values[planeSize * 2 + offset] else firstChannel

                val r = toColorComponent(firstChannel, normalization)
                val g = toColorComponent(secondChannel, normalization)
                val b = toColorComponent(thirdChannel, normalization)
                bitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }
    }

    private fun determineNormalization(values: FloatArray): NormalizationConfig {
        val min = values.minOrNull() ?: 0f
        val max = values.maxOrNull() ?: 0f

        val strategy = when {
            min >= 0f && max <= 1f -> NormalizationStrategy.ZERO_TO_ONE
            min >= -1f && max <= 1f -> NormalizationStrategy.MINUS_ONE_TO_ONE
            else -> NormalizationStrategy.MIN_MAX
        }

        return NormalizationConfig(strategy, min, max)
    }

    private fun toColorComponent(value: Float, normalization: NormalizationConfig): Int {
        val normalized = when (normalization.strategy) {
            NormalizationStrategy.ZERO_TO_ONE -> value.coerceIn(0f, 1f)
            NormalizationStrategy.MINUS_ONE_TO_ONE -> ((value + 1f) / 2f).coerceIn(0f, 1f)
            NormalizationStrategy.MIN_MAX -> if (normalization.max == normalization.min) {
                0.5f
            } else {
                ((value - normalization.min) / (normalization.max - normalization.min)).coerceIn(0f, 1f)
            }
        }

        return (normalized * 255f).roundToInt().coerceIn(0, 255)
    }

    private fun inferOutputDimensions(shape: IntArray): OutputDimensions? {
        if (shape.isEmpty()) {
            return null
        }

        val shapeWithoutBatch = if (shape.size > 1 && shape[0] == 1) {
            shape.copyOfRange(1, shape.size)
        } else {
            shape
        }

            return when (shapeWithoutBatch.size) {
                2 -> OutputDimensions(
                    width = shapeWithoutBatch[1],
                    height = shapeWithoutBatch[0],
                    channels = 1,
                    channelsLast = true,
                )

                3 -> {
                    val a = shapeWithoutBatch[0]
                    val b = shapeWithoutBatch[1]
                    val c = shapeWithoutBatch[2]

                    when {
                        c in 1..4 -> OutputDimensions(width = b, height = a, channels = c, channelsLast = true)
                        a in 1..4 -> OutputDimensions(width = c, height = b, channels = a, channelsLast = false)
                        else -> null
                    }
                }

                else -> null
            }
        }

        private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer? {
            return try {
                assetManager.open(filename, AssetManager.ACCESS_BUFFER).use { inputStream ->
                    val bytes = inputStream.readBytes()
                    ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                        put(bytes)
                        rewind()
                    }
                }
            } catch (fileNotFoundException: FileNotFoundException) {
                Log.w(TAG, "Generator model asset '$filename' not found. Falling back to procedural generator.")
                null
            } catch (ioException: IOException) {
                Log.e(TAG, "Unable to load generator model from assets", ioException)
                throw ioException
            }
        }

        private fun generateFallbackOutput(input: FloatArray): FloatArray {
            val dimensions = outputDimensions ?: return FloatArray(0)
            val totalPixels = dimensions.width * dimensions.height
            val channelCount = dimensions.channels
            val result = FloatArray(totalPixels * channelCount)

            val seed = input.fold(0L) { acc, value ->
                val scaled = (value * SEED_SCALING_FACTOR).toLong()
                acc xor scaled
            }
            val random = Random(seed)

            for (index in 0 until totalPixels) {
                val baseIndex = index * channelCount
                val brightness = random.nextFloat()
                val accent = random.nextFloat()
                result[baseIndex] = brightness
                if (channelCount > 1) {
                    result[baseIndex + 1] = (brightness + accent) / 2f
                }
                if (channelCount > 2) {
                    result[baseIndex + 2] = accent
                }
            }

            return result
        }

        fun isUsingFallbackModel(): Boolean = interpreter == null

        private data class OutputDimensions(
            val width: Int,
            val height: Int,
            val channels: Int,
            val channelsLast: Boolean,
        )

        private data class NormalizationConfig(
            val strategy: NormalizationStrategy,
            val min: Float,
            val max: Float,
        )

        private enum class NormalizationStrategy {
            ZERO_TO_ONE,
            MINUS_ONE_TO_ONE,
            MIN_MAX,
        }

        companion object {
            private const val TAG = "GeneratorApplicator"
            private const val MODEL_FILE_NAME = "model_dgenerator.tflite"
            private const val BYTES_PER_FLOAT = 4
            private const val FALLBACK_LATENT_VECTOR_LENGTH = 64
            private const val FALLBACK_IMAGE_WIDTH = 64
            private const val FALLBACK_IMAGE_HEIGHT = 64
            private const val FALLBACK_IMAGE_CHANNELS = 3
            private const val SEED_SCALING_FACTOR = 100_000
        }
    }