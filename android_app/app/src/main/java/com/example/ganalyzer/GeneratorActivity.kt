package com.example.ganalyzer

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.GeneratorApplicator
import com.google.android.material.bottomnavigation.BottomNavigationView
import java.io.IOException
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class GeneratorActivity : AppCompatActivity() {

    private var generatedValues: FloatArray? = null
    private var generatorApplicator: GeneratorApplicator? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_generator)

        Log.d(TAG, "onCreate: initializing GeneratorActivity")
        initializeGeneratorApplicator()
        setupBottomNavigation()
        setupButtons()
    }

    override fun onDestroy() {
        super.onDestroy()
        generatorApplicator?.close()
        generatorApplicator = null
    }

    private fun setupButtons() {
        val generateButton = findViewById<Button>(R.id.button_generate_array)
        val applyButton = findViewById<Button>(R.id.button_apply_generator)
        val generatedText = findViewById<TextView>(R.id.text_generated_array)
        val imagePreview = findViewById<ImageView>(R.id.image_generator_output)

        generateButton.setOnClickListener {
            val expectedSize = generatorApplicator?.expectedInputSize()?: ModelConfig.LATENT_SPACE_SIZE

            val random = java.util.Random()
            val values = FloatArray(expectedSize) { random.nextGaussian().toFloat()  }

            Log.d(TAG, "Generated new values: ${values.joinToString(limit = 5, truncated = "...")}")

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

            generatedValues = values
            applyButton.isEnabled = generatorApplicator != null
        }

        applyButton.setOnClickListener {
            val values = generatedValues
            if (values == null) {
                Log.w(TAG, "Apply button clicked before values were generated")
                Toast.makeText(this, R.string.generator_generate_first, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val applicator = generatorApplicator
            if (applicator == null) {
                Log.w(TAG, "GeneratorApplicator not ready when apply was requested")
                Toast.makeText(this, R.string.generator_model_not_ready, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            applyButton.isEnabled = false

            lifecycleScope.launch {
                Log.d(TAG, "Applying generator with ${'$'}{values.size} values")
                val bitmapResult = withContext(Dispatchers.Default) {
                    runCatching { applicator.applyToBitmap(values) }
                }

                applyButton.isEnabled = true

                bitmapResult.onSuccess { bitmap ->
                    Log.d(TAG, "Generator application succeeded: bitmap null? ${'$'}{bitmap == null}")
                    if (bitmap != null) {
                        Log.d("dE","ERROR 13 2")
                        imagePreview.setImageBitmap(bitmap)
                    } else {
                        Log.d("de","ERROR 13 1") // unable to interpret the generator output as an image
                        Toast.makeText(this@GeneratorActivity, R.string.generator_output_unexpected, Toast.LENGTH_SHORT).show()
                    }
                }.onFailure { throwable ->
                    Toast.makeText(
                        this@GeneratorActivity,
                        getString(R.string.generator_apply_failed, throwable.localizedMessage ?: throwable.toString()),
                        Toast.LENGTH_LONG,
                    ).show()
                }
            }
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

    companion object {
        private const val DEFAULT_GENERATED_VALUES = 64
        private const val TAG = "GeneratorActivity"
    }

    private fun initializeGeneratorApplicator() {
        if (generatorApplicator != null) {
            Log.d(TAG, "initializeGeneratorApplicator: already initialized")
            return
        }

        Log.d(TAG, "initializeGeneratorApplicator: creating GeneratorApplicator")
        val applicator = try {
            GeneratorApplicator(this)
        } catch (ioException: IOException) {
            Log.e(TAG, "Failed to load generator model", ioException)
            Toast.makeText(
                this,
                getString(R.string.generator_model_load_error, ioException.localizedMessage ?: ioException.toString()),
                Toast.LENGTH_LONG,
            ).show()
            null
        } catch (throwable: Throwable) {
            Log.e(TAG, "Unexpected error while loading generator model", throwable)
            Toast.makeText(
                this,
                getString(R.string.generator_model_load_error, throwable.localizedMessage ?: throwable.toString()),
                Toast.LENGTH_LONG,
            ).show()
            null
        }

        if (applicator != null && applicator.isUsingFallbackModel()) {
            Log.w(TAG, "GeneratorApplicator is using fallback model")
            Toast.makeText(this, R.string.generator_model_fallback_notice, Toast.LENGTH_LONG).show()
        }

        Log.d(TAG, "initializeGeneratorApplicator: applicator ${if (applicator == null) "not created" else "ready"}")

        generatorApplicator = applicator
    }
}