package com.example.hallucinator

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.DecoderApplicator
import com.example.EncoderApplicator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class EncoderAndDecoderActivity : AppCompatActivity() {
    private var encoderApplicator: EncoderApplicator? = null
    private var decoderApplicator: DecoderApplicator? = null
    private var modelInputBitmap: Bitmap? = null
    private var decoderOutputBitmap: Bitmap? = null

    private lateinit var inputPreview: ImageView
    private lateinit var outputPreview: ImageView
    private lateinit var selectImageButton: Button
    private lateinit var applyModelButton: Button
    private lateinit var statusText: TextView

    private val selectImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri == null) {
                return@registerForActivityResult
            }

            try {
                val originalBitmap = loadBitmapFromUri(uri)
                if (originalBitmap != null) {
                    val resizedBitmap = Bitmap.createScaledBitmap(
                        originalBitmap,
                        ModelConfig.MODEL_IMAGE_SIZE,
                        ModelConfig.MODEL_IMAGE_SIZE,
                        true
                    )
                    if (resizedBitmap != originalBitmap) {
                        originalBitmap.recycle()
                    }
                    modelInputBitmap?.recycle()
                    modelInputBitmap = resizedBitmap
                    decoderOutputBitmap?.recycle()
                    decoderOutputBitmap = null

                    inputPreview.setImageBitmap(resizedBitmap)
                    outputPreview.setImageDrawable(null)
                    statusText.setText(getString(R.string.status_ready_to_apply_model))
                    applyModelButton.isEnabled = true
                } else {
                    showError(getString(R.string.global_error))
                }
            } catch (error: IOException) {
                showError(
                    getString(R.string.error_io_exception) + error.localizedMessage ?: error.toString()
                )
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_decoder_and_encoder)

        inputPreview = findViewById(R.id.image_encoder_input)
        outputPreview = findViewById(R.id.image_decoder_output)
        selectImageButton = findViewById(R.id.button_select_image)
        applyModelButton = findViewById(R.id.button_apply_model)
        statusText = findViewById(R.id.text_status)

        selectImageButton.setOnClickListener { selectImageLauncher.launch("image/*") }
        applyModelButton.setOnClickListener { applyModels() }
        applyModelButton.isEnabled = false

        setupBottomNavigation(R.id.navigation_encoder_decoder)
    }

    override fun onDestroy() {
        encoderApplicator?.close()
        decoderApplicator?.close()
        encoderApplicator = null
        decoderApplicator = null

        modelInputBitmap?.recycle()
        modelInputBitmap = null
        decoderOutputBitmap?.recycle()
        decoderOutputBitmap = null

        super.onDestroy()
    }

    private fun applyModels() {
        val bitmap = modelInputBitmap
        if (bitmap == null) {
            showError(getString(R.string.error_no_image))
            return
        }

        val encoder = encoderApplicator ?: try {
            EncoderApplicator(this).also { encoderApplicator = it }
        } catch (error: Exception) {
            showError(getString(R.string.model_status_error, error.localizedMessage ?: error.toString()))
            return
        }

        val decoder = decoderApplicator ?: try {
            DecoderApplicator(this).also { decoderApplicator = it }
        } catch (error: Exception) {
            showError(getString(R.string.model_status_error, error.localizedMessage ?: error.toString()))
            return
        }

        applyModelButton.isEnabled = false
        statusText.text = getString(R.string.status_running_encoder)
        decoderOutputBitmap?.recycle()
        decoderOutputBitmap = null
        outputPreview.setImageDrawable(null)

        lifecycleScope.launch {
            try {
                val inputShape = encoder.inputShape
                val modelInput = createModelInputFromBitmap(bitmap, inputShape)

                val encoded = withContext(Dispatchers.Default) {
                    encoder.apply(modelInput)
                }
                statusText.text = getString(R.string.status_running_decoder)

                val decoded = withContext(Dispatchers.Default) {
                    decoder.apply(encoded)
                }

                val bitmapOutput = withContext(Dispatchers.Default) {
                    convertDecoderOutputToBitmap(decoded)
                }

                decoderOutputBitmap?.recycle()
                decoderOutputBitmap = bitmapOutput
                outputPreview.setImageBitmap(bitmapOutput)
                statusText.setText(getString(R.string.status_success))
            } catch (error: Exception) {
                decoderOutputBitmap?.recycle()
                decoderOutputBitmap = null
                outputPreview.setImageDrawable(null)
                val message = error.localizedMessage ?: error.toString()
                showError(getString(R.string.error_decode_model)+ message)
            } finally {
                applyModelButton.isEnabled = modelInputBitmap != null
            }
        }
    }

    private fun createModelInputFromBitmap(bitmap: Bitmap, inputShape: IntArray): FloatArray {
        val expectedSize = inputShape.fold(1) { acc, dimension -> acc * dimension }
        require(expectedSize > 0) { getString(R.string.error_invalid_input_shape, inputShape.contentToString()) }

        val batchSize = inputShape.firstOrNull() ?: 1
        require(batchSize == 1) { getString(R.string.wrong_batch_size, batchSize.toString()) }

        val (height, width, channels, channelFirst) = when {
            inputShape.size == 4 && inputShape[3] == 3 -> Quadruple(inputShape[1], inputShape[2], inputShape[3], false)
            inputShape.size == 4 && inputShape[1] == 3 -> Quadruple(inputShape[2], inputShape[3], inputShape[1], true)
            inputShape.size == 3 && inputShape[2] == 3 -> Quadruple(inputShape[0], inputShape[1], inputShape[2], false)
            inputShape.size == 3 && inputShape[0] == 3 -> Quadruple(inputShape[1], inputShape[2], inputShape[0], true)
            else -> throw IllegalArgumentException(getString(R.string.unsupported_input_shape, inputShape.contentToString()))
        }

        require(channels == 3) { getString(R.string.wrong_channels_qty, channels.toString()) }
        require(height == ModelConfig.MODEL_IMAGE_SIZE && width == ModelConfig.MODEL_IMAGE_SIZE) {
            getString(R.string.model_expects_x_but_activity_resizes_to_x, height.toString(), width.toString(), ModelConfig.MODEL_IMAGE_SIZE.toString(), ModelConfig.MODEL_IMAGE_SIZE.toString())
        }

        val pixels = IntArray(ModelConfig.MODEL_IMAGE_SIZE * ModelConfig.MODEL_IMAGE_SIZE)
        bitmap.getPixels(
            pixels,
            0,
            ModelConfig.MODEL_IMAGE_SIZE,
            0,
            0,
            ModelConfig.MODEL_IMAGE_SIZE,
            ModelConfig.MODEL_IMAGE_SIZE
        )

        val output = FloatArray(expectedSize)
        val normalizationFactor = 1f / 255f
        var pixelIndex = 0

        if (channelFirst) {
            val channelSize = ModelConfig.MODEL_IMAGE_SIZE * ModelConfig.MODEL_IMAGE_SIZE
            for (y in 0 until ModelConfig.MODEL_IMAGE_SIZE) {
                for (x in 0 until ModelConfig.MODEL_IMAGE_SIZE) {
                    val pixel = pixels[pixelIndex++]
                    val r = ((pixel shr 16) and 0xFF) * normalizationFactor
                    val g = ((pixel shr 8) and 0xFF) * normalizationFactor
                    val b = (pixel and 0xFF) * normalizationFactor

                    val baseIndex = y * ModelConfig.MODEL_IMAGE_SIZE + x
                    output[baseIndex] = r
                    output[channelSize + baseIndex] = g
                    output[channelSize * 2 + baseIndex] = b
                }
            }
        } else {
            var outputIndex = 0
            for (y in 0 until ModelConfig.MODEL_IMAGE_SIZE) {
                for (x in 0 until ModelConfig.MODEL_IMAGE_SIZE) {
                    val pixel = pixels[pixelIndex++]
                    output[outputIndex++] = ((pixel shr 16) and 0xFF) * normalizationFactor
                    output[outputIndex++] = ((pixel shr 8) and 0xFF) * normalizationFactor
                    output[outputIndex++] = (pixel and 0xFF) * normalizationFactor
                }
            }
        }

        return output
    }

    private fun convertDecoderOutputToBitmap(values: FloatArray): Bitmap {
        require(values.size == ModelConfig.DECODER_OUTPUT_SIZE) {
            getString(R.string.decoder_output_size_does_not_match_expected, values.size.toString(), ModelConfig.DECODER_OUTPUT_SIZE.toString())
        }

        val pixels = IntArray(ModelConfig.DECODER_IMAGE_WIDTH * ModelConfig.DECODER_IMAGE_HEIGHT)
        var index = 0
        for (position in pixels.indices) {
            val red = convertChannel(values[index++])
            val green = convertChannel(values[index++])
            val blue = convertChannel(values[index++])
            pixels[position] = Color.rgb(red, green, blue)
        }

        return Bitmap.createBitmap(
            pixels,
            ModelConfig.DECODER_IMAGE_WIDTH,
            ModelConfig.DECODER_IMAGE_HEIGHT,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun convertChannel(value: Float): Int {
        val scaled = (value.coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        return scaled.coerceIn(0, 255)
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    }

    private fun showError(message: String) {
        statusText.text = getString(R.string.global_error_2)+ message
        Log.e(TAG, message)
    }

    private data class Quadruple(
        val first: Int,
        val second: Int,
        val third: Int,
        val fourth: Boolean
    )

    companion object {
        private const val TAG = "EncoderDecoderActivity"
    }
}