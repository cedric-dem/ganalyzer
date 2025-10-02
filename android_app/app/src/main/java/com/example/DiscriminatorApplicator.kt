package com.example

import android.content.Context
import android.content.res.AssetManager
import com.example.ganalyzer.ModelConfig
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class DiscriminatorApplicator(context: Context) : Closeable {

    private val interpreter: Interpreter
    private val inputElementCount: Int
    private val outputElementCount: Int

    val inputImageWidth: Int
    val inputImageHeight: Int
    val inputChannels: Int

    init {
        val modelBuffer = loadModelFile(context.assets, ModelConfig.DISCRIMINATOR_PATH)
        interpreter = Interpreter(modelBuffer)

        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        require(inputShape.isNotEmpty()) { "Model does not expose an input tensor" }
        require(inputShape[0] == 1) { "Only batch size of 1 is supported but was ${inputShape[0]}" }
        require(inputShape.size == 4) { "Unsupported input shape: ${inputShape.contentToString()}" }

        inputImageHeight = inputShape[1]
        inputImageWidth = inputShape[2]
        inputChannels = inputShape[3]

        require(inputChannels == ModelConfig.DECODER_IMAGE_CHANNELS) {
            "Expected ${ModelConfig.DECODER_IMAGE_CHANNELS} channels but was $inputChannels"
        }

        inputElementCount = inputTensor.numElements() / inputShape[0]
        outputElementCount = interpreter.getOutputTensor(0).numElements()
    }

    fun apply(input: FloatArray): FloatArray {
        require(input.size == inputElementCount) {
            "Invalid input length ${input.size}. Expected $inputElementCount elements"
        }

        val inputBuffer = ByteBuffer.allocateDirect(4 * input.size).order(ByteOrder.nativeOrder())
        val inputFloatView = inputBuffer.asFloatBuffer()
        inputFloatView.put(input)
        inputBuffer.rewind()

        val outputBuffer = ByteBuffer.allocateDirect(4 * outputElementCount).order(ByteOrder.nativeOrder())
        val outputFloatView = outputBuffer.asFloatBuffer()

        interpreter.run(inputBuffer, outputBuffer)

        outputFloatView.rewind()
        val output = FloatArray(outputElementCount)
        outputFloatView.get(output)
        return output
    }

    override fun close() {
        interpreter.close()
    }

    private fun loadModelFile(assetManager: AssetManager, path: String): ByteBuffer {
        return assetManager.open(path).use { inputStream ->
            val bytes = readAllBytes(inputStream)
            ByteBuffer.allocateDirect(bytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(bytes)
                rewind()
            }
        }
    }

    private fun readAllBytes(inputStream: InputStream): ByteArray {
        val outputStream = ByteArrayOutputStream()
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        var read: Int
        while (true) {
            read = inputStream.read(buffer)
            if (read == -1) {
                break
            }
            outputStream.write(buffer, 0, read)
        }
        return outputStream.toByteArray()
    }

    companion object {
        private const val DEFAULT_BUFFER_SIZE = 16 * 1024
    }
}