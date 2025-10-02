package com.example.hallucinator


import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.DecoderApplicator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.random.Random

class OnlyDecoderActivity : AppCompatActivity() {
    private var decoderApplicator: DecoderApplicator? = null
    private var currentInput: FloatArray? = null


    private fun getPseudoRandom(): Float {
        val intervals = listOf(
            0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.0f,0.0f to 0.01f,0.01f to 0.03f,0.03f to 0.05f,0.05f to 0.07f,0.07f to 0.09f,0.09f to 0.11f,0.11f to 0.13f,0.13f to 0.15f,0.15f to 0.17f,0.17f to 0.19f,0.19f to 0.21f,0.21f to 0.23f,0.23f to 0.25f,0.25f to 0.27f,0.27f to 0.3f,0.3f to 0.32f,0.32f to 0.34f,0.34f to 0.36f,0.36f to 0.38f,0.38f to 0.4f,0.4f to 0.42f,0.42f to 0.44f,0.44f to 0.46f,0.46f to 0.48f,0.48f to 0.5f,0.5f to 0.52f,0.52f to 0.55f,0.55f to 0.57f,0.57f to 0.59f,0.59f to 0.61f,0.61f to 0.63f,0.63f to 0.65f,0.65f to 0.67f,0.67f to 0.7f,0.7f to 0.72f,0.72f to 0.74f,0.74f to 0.76f,0.76f to 0.78f,0.78f to 0.81f,0.81f to 0.83f,0.83f to 0.85f,0.85f to 0.88f,0.88f to 0.9f,0.9f to 0.92f,0.92f to 0.95f,0.95f to 0.97f,0.97f to 0.99f,0.99f to 1.02f,1.02f to 1.04f,1.04f to 1.07f,1.07f to 1.09f,1.09f to 1.11f,1.11f to 1.14f,1.14f to 1.16f,1.16f to 1.18f,1.18f to 1.21f,1.21f to 1.23f,1.23f to 1.26f,1.26f to 1.28f,1.28f to 1.3f,1.3f to 1.33f,1.33f to 1.35f,1.35f to 1.38f,1.38f to 1.4f,1.4f to 1.43f,1.43f to 1.45f,1.45f to 1.48f,1.48f to 1.5f,1.5f to 1.53f,1.53f to 1.55f,1.55f to 1.58f,1.58f to 1.6f,1.6f to 1.63f,1.63f to 1.66f,1.66f to 1.68f,1.68f to 1.71f,1.71f to 1.73f,1.73f to 1.76f,1.76f to 1.79f,1.79f to 1.81f,1.81f to 1.84f,1.84f to 1.87f,1.87f to 1.9f,1.9f to 1.92f,1.92f to 1.95f,1.95f to 1.98f,1.98f to 2.01f,2.01f to 2.03f,2.03f to 2.06f,2.06f to 2.09f,2.09f to 2.12f,2.12f to 2.15f,2.15f to 2.17f,2.17f to 2.2f,2.2f to 2.23f,2.23f to 2.26f,2.26f to 2.29f,2.29f to 2.32f,2.32f to 2.35f,2.35f to 2.38f,2.38f to 2.41f,2.41f to 2.44f,2.44f to 2.47f,2.47f to 2.5f,2.5f to 2.53f,2.53f to 2.56f,2.56f to 2.59f,2.59f to 2.62f,2.62f to 2.65f,2.65f to 2.68f,2.68f to 2.71f,2.71f to 2.74f,2.74f to 2.78f,2.78f to 2.81f,2.81f to 2.84f,2.84f to 2.87f,2.87f to 2.91f,2.91f to 2.94f,2.94f to 2.98f,2.98f to 3.01f,3.01f to 3.05f,3.05f to 3.09f,3.09f to 3.13f,3.13f to 3.17f,3.17f to 3.2f,3.2f to 3.25f,3.25f to 3.29f,3.29f to 3.33f,3.33f to 3.37f,3.37f to 3.42f,3.42f to 3.47f,3.47f to 3.52f,3.52f to 3.57f,3.57f to 3.62f,3.62f to 3.68f,3.68f to 3.73f,3.73f to 3.79f,3.79f to 3.85f,3.85f to 3.92f,3.92f to 3.99f,3.99f to 4.07f,4.07f to 4.15f,4.15f to 4.25f,4.25f to 4.36f,4.36f to 4.5f,4.5f to 4.69f,4.69f to 5.02f,5.02f to 7.52f,
        )

        val chosen = intervals.random()

        return if (chosen.first == chosen.second) {
            chosen.first
        } else {
            Random.nextFloat() * (chosen.second - chosen.first) + chosen.first
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_only_decoder)

        setupBottomNavigation(R.id.navigation_only_decoder)

        val generateButton = findViewById<Button>(R.id.button_generate_array)
        val applyButton = findViewById<Button>(R.id.button_apply_decoder)
        val generatedArrayText = findViewById<TextView>(R.id.text_generated_array)
        val decoderOutputImage = findViewById<ImageView>(R.id.image_decoder_output)

        decoderApplicator = try {
            DecoderApplicator(this)
        } catch (error: Exception) {
            val message = error.localizedMessage ?: getString(R.string.unknown_error)+ error.toString()
            Toast.makeText(this, getString(R.string.model_status_error, message), Toast.LENGTH_LONG).show()
            null
        }

        generateButton.setOnClickListener {
            val array = FloatArray(ModelConfig.DECODER_INPUT_SIZE) { getPseudoRandom() }
            currentInput = array
            generatedArrayText.text = formatFloatArray(array)
            decoderOutputImage.setImageDrawable(null)
        }

        applyButton.setOnClickListener {
            val applicator = decoderApplicator
            if (applicator == null) {
                Toast.makeText(
                    this,
                    getString(R.string.model_status_error, "decoder out error"),
                    Toast.LENGTH_LONG
                ).show()
                return@setOnClickListener
            }

            val input = currentInput
            if (input == null) {
                Toast.makeText(this, getString(R.string.input_here), Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            applyButton.isEnabled = false
            decoderOutputImage.setImageDrawable(null)

            lifecycleScope.launch {
                try {
                    val result = withContext(Dispatchers.Default) {
                        applicator.apply(input)
                    }
                    val bitmap = convertDecoderOutputToBitmap(result)
                    decoderOutputImage.setImageBitmap(bitmap)
                } catch (error: Exception) {
                    decoderOutputImage.setImageDrawable(null)
                    val message = error.localizedMessage ?: getString(R.string.decoder_error)
                    Toast.makeText(
                        this@OnlyDecoderActivity,
                        getString(R.string.apply_error) +  message,
                        Toast.LENGTH_LONG
                    ).show()
                } finally {
                    applyButton.isEnabled = true
                }
            }
        }
    }

    override fun onDestroy() {
        decoderApplicator?.close()
        decoderApplicator = null
        super.onDestroy()
    }

    private fun formatFloatArray(values: FloatArray): String {
        return values.joinToString(prefix = "[", postfix = "]", separator = ", ") {
            String.format(Locale.US, "%.4f", it)
        }
    }

    private fun convertDecoderOutputToBitmap(values: FloatArray): Bitmap {
        require(values.size == ModelConfig.DECODER_OUTPUT_SIZE) {
            getString(R.string.decoder_output_size_does_not_match_expected_2, values.size, ModelConfig.DECODER_OUTPUT_SIZE)
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
}