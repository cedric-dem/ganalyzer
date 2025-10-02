package com.example

import android.content.Context
import android.content.res.AssetManager
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class DecoderApplicator(context: Context) : Closeable {
    private val interpreter: Interpreter
    private var configuredInputSize: Int? = null

    init {
        this.interpreter = Interpreter(loadModelFile(context.assets, MODEL_ASSET_NAME))
    }

    fun apply(input: FloatArray): FloatArray {
        ensureInterpreterConfigured(input.size)

        val expectedSize = configuredInputSize
        require(expectedSize != null && input.size == expectedSize) {
            "Input has " + input.size + " elements but expected " + expectedSize
        }

        val inputBuffer = ByteBuffer
            .allocateDirect(java.lang.Float.BYTES * input.size)
            .order(ByteOrder.nativeOrder())
        inputBuffer.asFloatBuffer().put(input)
        inputBuffer.rewind()

        val outputTensor = interpreter.getOutputTensor(0)

        val outputShape = outputTensor.shape()
        val outputContainer = allocateOutputContainer(outputShape)

        interpreter.run(inputBuffer, outputContainer)

        val floatResult = flattenOutput(outputContainer, outputShape)
        return floatResult
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        private const val MODEL_ASSET_NAME = "model_decoder.tflite"

        private fun numElements(shape: IntArray): Int {
            var product = 1
            for (dimension in shape) {
                if (dimension <= 0) continue
                product *= dimension
            }
            return product
        }

        @Throws(IOException::class)
        private fun loadModelFile(assetManager: AssetManager, assetName: String): ByteBuffer {
            assetManager.open(assetName).use { inputStream ->
                val modelBytes = readAllBytes(inputStream)
                val buffer = ByteBuffer
                    .allocateDirect(modelBytes.size)
                    .order(ByteOrder.nativeOrder())
                buffer.put(modelBytes)
                buffer.rewind()
                return buffer
            }
        }

        @Throws(IOException::class)
        private fun readAllBytes(inputStream: InputStream): ByteArray {
            val outputStream = ByteArrayOutputStream()
            val chunk = ByteArray(8192)
            var bytesRead: Int
            while ((inputStream.read(chunk).also { bytesRead = it }) != -1) {
                outputStream.write(chunk, 0, bytesRead)
            }
            return outputStream.toByteArray()
        }
    }

    private fun ensureInterpreterConfigured(inputSize: Int) {
        if (configuredInputSize == inputSize) {
            return
        }

        interpreter.resizeInput(0, intArrayOf(1, inputSize))
        interpreter.allocateTensors()

        val updatedShape = interpreter.getInputTensor(0).shape()
        configuredInputSize = numElements(updatedShape)
    }

    private fun allocateOutputContainer(shape: IntArray): Any {
        require(shape.isNotEmpty()) { "Output tensor has no shape" }
        return when (shape.size) {
            1 -> FloatArray(shape[0])
            2 -> Array(shape[0]) { FloatArray(shape[1]) }
            3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
            4 -> Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } } }
            else -> throw IllegalArgumentException("Unsupported output tensor shape: " + shape.contentToString())
        }
    }

    private fun flattenOutput(container: Any, shape: IntArray): FloatArray {
        val result = FloatArray(numElements(shape))
        var position = 0

        fun append(value: Any?) {
            when (value) {
                is FloatArray -> {
                    value.copyInto(result, position)
                    position += value.size
                }
                is Array<*> -> {
                    for (element in value) {
                        append(element)
                    }
                }
                null -> throw IllegalArgumentException("Output container contains null values")
                else -> throw IllegalArgumentException("Unsupported element type in output container: ${value::class.java.simpleName}")
            }
        }

        append(container)
        return result
    }
}