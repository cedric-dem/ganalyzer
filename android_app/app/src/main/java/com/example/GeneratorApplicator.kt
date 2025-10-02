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

class GeneratorApplicator(context: Context) : Closeable {

    init {
        //TODO
    }

    fun apply(input: FloatArray): FloatArray? {
        //TODO
        return null
    }

    override fun close() {
        TODO("Not yet implemented")
    }
}