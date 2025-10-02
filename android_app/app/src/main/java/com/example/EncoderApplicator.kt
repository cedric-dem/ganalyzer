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

class EncoderApplicator(context: Context) : Closeable {
    private val interpreter: Interpreter

    init {
        this.interpreter = Interpreter(loadModelFile(context.assets, MODEL_ASSET_NAME))
    }

    fun apply(input: FloatArray): FloatArray {
        val inputShape = inputShape
        val expectedSize = numElements(inputShape)
        require(input.size == expectedSize) { "Input has " + input.size + " elements but expected " + expectedSize }

        val inputBuffer = ByteBuffer
            .allocateDirect(java.lang.Float.BYTES * input.size)
            .order(ByteOrder.nativeOrder())
        for (value in input) {
            inputBuffer.putFloat(value)
        }
        inputBuffer.rewind()

        val outputShape = outputShape
        val outputSize = numElements(outputShape)
        val outputBuffer = ByteBuffer
            .allocateDirect(java.lang.Float.BYTES * outputSize)
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val floatBuffer = outputBuffer.asFloatBuffer()
        val output = FloatArray(outputSize)
        floatBuffer.get(output)

        return output
    }

    val inputShape: IntArray
        get() = interpreter.getInputTensor(0).shape().clone()

    val outputShape: IntArray
        get() = interpreter.getOutputTensor(0).shape().clone()

    override fun close() {
        interpreter.close()
    }

    companion object {
        private const val MODEL_ASSET_NAME = "model_encoder.tflite"

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

        private fun numElements(shape: IntArray): Int {
            var product = 1
            for (dimension in shape) {
                product *= dimension
            }
            return product
        }
    }
}