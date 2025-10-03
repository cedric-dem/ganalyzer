from __future__ import annotations

import pathlib
from typing import Iterable

import tensorflow as tf


def _load_model(model_path: pathlib.Path) -> tf.keras.Model:
    """Load a compiled Keras model from disk."""

    return tf.keras.models.load_model(str(model_path), compile=False)


def _build_concrete_function(model: tf.keras.Model) -> tf.types.experimental.ConcreteFunction:
    """Create a concrete function compatible with TFLite conversion."""

    input_specs: list[tf.TensorSpec] = []
    for tensor in model.inputs:
        # Replace unknown batch dimensions with a concrete value so the
        # converter can materialize a ConcreteFunction.
        shape = [dim if dim is not None else 1 for dim in tensor.shape]
        input_specs.append(tf.TensorSpec(shape=shape, dtype=tensor.dtype))

    @tf.function
    def model_fn(*args):
        return model(*args, training = False)

    return model_fn.get_concrete_function(*input_specs)


def _configure_converter(
    concrete_fn: tf.types.experimental.ConcreteFunction, model: tf.keras.Model
) -> tf.lite.TFLiteConverter:
    """Create a converter tuned for legacy-mobile compatibility."""

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)

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
    concrete_function = _build_concrete_function(model)
    converter = _configure_converter(concrete_function, model)
    tflite_model = converter.convert()
    model_path_tflite.write_bytes(tflite_model)


def _default_models() -> Iterable[tuple[pathlib.Path, pathlib.Path]]:
    root = pathlib.Path(__file__).resolve().parent
    return (
        (root / "generator.keras", root / "generator.tflite"),
        (root / "discriminator.keras", root / "discriminator.tflite"),
    )


def main() -> None:
    for source, target in _default_models():
        print(f"Converting {source.name} -> {target.name}")
        export_tflite(source, target)


if __name__ == "__main__":
    main()