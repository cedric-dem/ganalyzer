package com.example.hallucinator

import android.content.Intent
import androidx.annotation.IdRes
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.bottomnavigation.BottomNavigationView

fun AppCompatActivity.setupBottomNavigation(@IdRes selectedItemId: Int) {
    val bottomNavigationView: BottomNavigationView = findViewById(R.id.bottom_navigation)
    bottomNavigationView.selectedItemId = selectedItemId

    bottomNavigationView.setOnItemSelectedListener { item ->
        if (item.itemId == selectedItemId) {
            true
        } else {
            when (item.itemId) {
                R.id.navigation_main -> {
                    startActivity(Intent(this, OnlyEncoderActivity::class.java).apply {
                        addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
                    })
                    true
                }
                R.id.navigation_only_decoder -> {
                    startActivity(Intent(this, OnlyDecoderActivity::class.java).apply {
                        addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
                    })
                    true
                }
                R.id.navigation_encoder_decoder -> {
                    startActivity(Intent(this, EncoderAndDecoderActivity::class.java).apply {
                        addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
                    })
                    true
                }
                else -> false
            }
        }
    }

    bottomNavigationView.setOnItemReselectedListener { }
}