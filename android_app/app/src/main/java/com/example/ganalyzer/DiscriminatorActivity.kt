package com.example.ganalyzer

import android.content.Intent
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
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.DiscriminatorApplicator
import com.google.android.material.bottomnavigation.BottomNavigationView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class DiscriminatorActivity : AppCompatActivity() {
    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            handleSelectedImage(uri)
        }
    }

    private lateinit var selectImageButton: Button
    private lateinit var applyButton: Button
    private lateinit var previewImage: ImageView
    private lateinit var predictionText: TextView

    private var discriminatorApplicator: DiscriminatorApplicator? = null
    private var selectedBitmap: Bitmap? = null
    private var modelLoadFailed: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_discriminator)

        previewImage = findViewById(R.id.discriminator_input_preview)
        predictionText = findViewById(R.id.discriminator_text_prediction_output)
        selectImageButton = findViewById(R.id.button_select_image)
        applyButton = findViewById(R.id.button_apply_discriminator)

        predictionText.text = getString(R.string.status_loading_discriminator_model)
        applyButton.isEnabled = false

        setupBottomNavigation()
        setupButtons()
        loadDiscriminatorModel()
    }

    override fun onDestroy() {
        super.onDestroy()
        discriminatorApplicator?.close()
        discriminatorApplicator = null
        selectedBitmap?.recycle()
        selectedBitmap = null
    }

    private fun setupButtons() {
        selectImageButton.setOnClickListener {
            imagePicker.launch("image/*")
        }

        applyButton.setOnClickListener {
            onApplyClicked()
        }
    }

    private fun setupBottomNavigation() {
        val bottomNavigation = findViewById<BottomNavigationView>(R.id.bottom_navigation)
        bottomNavigation.selectedItemId = R.id.navigation_only_decoder
        bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.navigation_main -> {
                    startActivity(Intent(this, GeneratorActivity::class.java))
                    overridePendingTransition(0, 0)
                    finish()
                    true
                }

                R.id.navigation_only_decoder -> true
                else -> false
            }
        }
    }

    private fun loadDiscriminatorModel() {
        modelLoadFailed = false
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { DiscriminatorApplicator(this@DiscriminatorActivity) }
            }
            result.onSuccess { applicator ->
                modelLoadFailed = false
                discriminatorApplicator = applicator
                if (selectedBitmap != null) {
                    predictionText.text = getString(R.string.status_ready_to_apply_model)
                } else {
                    predictionText.text = getString(R.string.status_waiting_for_image_selection)
                }
                updateApplyButtonState()
            }.onFailure { throwable ->
                modelLoadFailed = true
                Log.e(TAG, "Error loading discriminator model", throwable)
                predictionText.text = getString(R.string.error_loading_model)
                Toast.makeText(
                    this@DiscriminatorActivity,
                    R.string.error_loading_model,
                    Toast.LENGTH_SHORT
                ).show()
                updateApplyButtonState()
            }
        }
    }

    private fun handleSelectedImage(uri: Uri) {
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { loadBitmap(uri) }
            }
            result.onSuccess { bitmap ->
                val previousBitmap = selectedBitmap
                selectedBitmap = bitmap
                previewImage.setImageBitmap(bitmap)
                previousBitmap?.recycle()
                predictionText.text = when {
                    discriminatorApplicator != null -> getString(R.string.status_ready_to_apply_model)
                    modelLoadFailed -> getString(R.string.error_loading_model)
                    else -> getString(R.string.status_loading_discriminator_model)
                }
                updateApplyButtonState()
            }.onFailure { ioException ->
                Log.e(TAG, "Error loading selected image", ioException)
                Toast.makeText(this@DiscriminatorActivity, R.string.error_loading_image, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun onApplyClicked() {
        val bitmap = selectedBitmap
        val applicator = discriminatorApplicator

        if (bitmap == null || applicator == null) {
            Toast.makeText(this, R.string.apply_the_model_to_see_output, Toast.LENGTH_SHORT).show()
            return
        }

        applyButton.isEnabled = false
        predictionText.text = getString(R.string.status_running_discriminator)

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val input = preprocessBitmap(bitmap, applicator.inputImageWidth, applicator.inputImageHeight, applicator.inputChannels)
                    applicator.apply(input)
                }
            }

            result.onSuccess { output ->
                if (output.isNotEmpty()) {
                    val prediction = output.first()
                    predictionText.text = getString(
                        R.string.discriminator_prediction_format,
                        String.format(Locale.US, "%.4f", prediction)
                    )
                } else {
                    predictionText.text = getString(R.string.error_applying_discriminator)
                }
            }.onFailure { throwable ->
                Log.e(TAG, "Error applying discriminator", throwable)
                predictionText.text = getString(R.string.error_applying_discriminator)
                Toast.makeText(
                    this@DiscriminatorActivity,
                    R.string.error_applying_discriminator,
                    Toast.LENGTH_SHORT
                ).show()
            }

            updateApplyButtonState()
        }
    }

    private fun updateApplyButtonState() {
        applyButton.isEnabled = selectedBitmap != null && discriminatorApplicator != null
    }

    @Suppress("DEPRECATION")
    private fun loadBitmap(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source)
        } else {
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    }

    private fun preprocessBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int, channels: Int): FloatArray {
        require(channels == ModelConfig.DECODER_IMAGE_CHANNELS) {
            "Unsupported channel count: $channels"
        }

        val scaledBitmap = if (bitmap.width == targetWidth && bitmap.height == targetHeight) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        }

        val bitmapForProcessing = if (scaledBitmap.config == Bitmap.Config.ARGB_8888) {
            scaledBitmap
        } else {
            scaledBitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        val result = FloatArray(targetWidth * targetHeight * channels)
        var index = 0
        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val color = bitmapForProcessing.getPixel(x, y)
                result[index++] = normalizeChannel(Color.red(color))
                result[index++] = normalizeChannel(Color.green(color))
                result[index++] = normalizeChannel(Color.blue(color))
            }
        }

        if (bitmapForProcessing !== scaledBitmap && bitmapForProcessing !== bitmap) {
            bitmapForProcessing.recycle()
        }

        if (scaledBitmap !== bitmap) {
            scaledBitmap.recycle()
        }

        return result
    }

    private fun normalizeChannel(value: Int): Float {
        return (value / 127.5f) - 1f
    }

    companion object {
        private const val TAG = "DiscriminatorActivity"
    }
}