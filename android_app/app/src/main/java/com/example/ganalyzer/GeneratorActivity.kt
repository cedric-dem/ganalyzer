package com.example.ganalyzer

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.bottomnavigation.BottomNavigationView
import java.util.Locale
import kotlin.math.sqrt
import kotlin.random.Random

class GeneratorActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_generator)

        setupBottomNavigation()
        setupButtons()
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    private fun setupButtons() {
        val generateButton = findViewById<Button>(R.id.button_generate_array)
        val applyButton = findViewById<Button>(R.id.button_apply_generator)
        val generatedText = findViewById<TextView>(R.id.text_generated_array)
        val imagePreview = findViewById<ImageView>(R.id.image_generator_output)

        generateButton.setOnClickListener {
            val values = FloatArray(DEFAULT_GENERATED_VALUES) { Random.nextFloat() }

            val builder = StringBuilder()
            builder.append('[')
            values.forEachIndexed { index, value ->
                builder.append(String.format(Locale.US, "%.2f", value))
                if (index != values.lastIndex) {
                    builder.append(", ")
                }
            }
            builder.append(']')
            generatedText.text = builder.toString()

            applyButton.isEnabled = true
            imagePreview.setImageBitmap(createPreviewBitmap(values))
        }

        applyButton.setOnClickListener {
            Toast.makeText(this, R.string.apply_the_model_to_see_output, Toast.LENGTH_SHORT).show()
        }

        applyButton.isEnabled = false
    }

    private fun setupBottomNavigation() {
        val bottomNavigation = findViewById<BottomNavigationView>(R.id.bottom_navigation)
        bottomNavigation.selectedItemId = R.id.navigation_main
        bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.navigation_only_decoder -> {
                    startActivity(Intent(this, DiscriminatorActivity::class.java))
                    overridePendingTransition(0, 0)
                    finish()
                    true
                }

                R.id.navigation_main -> true
                else -> false
            }
        }
    }

    private fun createPreviewBitmap(values: FloatArray): Bitmap {
        val size = sqrt(values.size.toFloat()).toInt().coerceAtLeast(1)
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        values.forEachIndexed { index, value ->
            val x = index % size
            val y = index / size
            val channel = (value.coerceIn(0f, 1f) * 255).toInt()
            bitmap.setPixel(x, y, Color.rgb(channel, channel, channel))
        }
        return bitmap
    }

    companion object {
        private const val DEFAULT_GENERATED_VALUES = 64
    }
}