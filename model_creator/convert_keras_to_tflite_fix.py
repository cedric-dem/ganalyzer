
"""TensorFlow Lite conversion utilities for mobile deployment.

This script converts the trained Keras generator and discriminator models
into TensorFlow Lite (TFLite) models that are compatible with older TFLite
runtime binaries (such as those bundled with many Android releases).

The generated models avoid newer builtin operation versions (e.g.
`FULLY_CONNECTED` v12) that can trigger runtime errors like::

    java.lang.IllegalArgumentException: Internal error: Cannot create
    interpreter: Didn't find op for builtin opcode 'FULLY_CONNECTED'
    version '12'.

Usage:
    python convert_to_tflite.py

The resulting `.tflite` files are written next to the source `.keras`
models.
"""

from __future__ import annotations

import pathlib
from typing import Iterable

import tensorflow as tf


def _load_model(model_path: pathlib.Path) -> tf.keras.Model:
    """Load a compiled Keras model from disk."""

    return tf.keras.models.load_model(str(model_path), compile=False)


def _configure_converter(model: tf.keras.Model) -> tf.lite.TFLiteConverter:
    """Create a converter tuned for legacy-mobile compatibility."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Disable aggressive optimizations that introduce newer operator
    # versions. This keeps the generated model compatible with older
    # Android TensorFlow Lite runtimes.
    converter.experimental_new_converter = False
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = False

    # Keep the model in float32 to avoid quantization induced operator
    # upgrades.
    converter.optimizations = []

    return converter


def export_tflite(model_path_keras: pathlib.Path, model_path_tflite: pathlib.Path) -> None:
    """Convert a Keras model to the TensorFlow Lite format."""

    model = _load_model(model_path_keras)
    converter = _configure_converter(model)
    tflite_model = converter.convert()
    model_path_tflite.write_bytes(tflite_model)


def _default_models() -> Iterable[tuple[pathlib.Path, pathlib.Path]]:
    root = pathlib.Path(__file__).resolve().parent
    return (
        (root / "model_generator.keras", root / "model_generator_legacy.tflite"),
        (root / "model_discriminator.keras", root / "model_discriminator_legacy.tflite"),
    )


def main() -> None:
    for source, target in _default_models():
        print(f"Converting {source.name} -> {target.name}")
        export_tflite(source, target)


if __name__ == "__main__":
    main()