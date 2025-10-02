package com.example.ganalyzer

import android.content.Intent
import android.graphics.Bitmap
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
import com.google.android.material.bottomnavigation.BottomNavigationView
import java.io.IOException

class DiscriminatorActivity : AppCompatActivity() {
    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            handleSelectedImage(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_discriminator)

        setupBottomNavigation()
        setupButtons()
        findViewById<TextView>(R.id.discriminator_text_prediction_output).text =
            getString(R.string.status_waiting_for_image_selection)
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    private fun setupButtons() {
        val selectImageButton = findViewById<Button>(R.id.button_select_image)
        val applyButton = findViewById<Button>(R.id.button_apply_discriminator)

        selectImageButton.setOnClickListener {
            imagePicker.launch("image/*")
        }

        applyButton.setOnClickListener {
            Toast.makeText(this, R.string.apply_the_model_to_see_output, Toast.LENGTH_SHORT).show()
        }

        applyButton.isEnabled = false
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

    private fun handleSelectedImage(uri: Uri) {
        try {
            val bitmap = loadBitmap(uri)
            findViewById<ImageView>(R.id.discriminator_input_preview).setImageBitmap(bitmap)
            findViewById<Button>(R.id.button_apply_discriminator).isEnabled = true
            findViewById<TextView>(R.id.discriminator_text_prediction_output).text =
                getString(R.string.status_ready_to_apply_model)
        } catch (ioException: IOException) {
            Log.e(TAG, "Error loading selected image", ioException)
            Toast.makeText(this, R.string.error_loading_image, Toast.LENGTH_SHORT).show()
        }
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

    companion object {
        private const val TAG = "DiscriminatorActivity"
    }
}