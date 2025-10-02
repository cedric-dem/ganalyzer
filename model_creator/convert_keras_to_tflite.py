import tensorflow as tf

def _build_concrete_function(model: tf.keras.Model) -> tf.types.experimental.ConcreteFunction:
    input_specs = []
    for tensor in model.inputs:
        shape = [dim if dim is not None else 1 for dim in tensor.shape]
        input_specs.append(tf.TensorSpec(shape=shape, dtype=tensor.dtype))

    @tf.function
    def model_fn(*args):
        return model(*args)

    return model_fn.get_concrete_function(*input_specs)


def export_tflite(model_path_keras: str, model_path_tflite: str) -> None:
    model = tf.keras.models.load_model(model_path_keras, compile=False)

    concrete_function = _build_concrete_function(model)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_function], model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    with open(model_path_tflite, "wb") as f:
        f.write(tflite_model)

export_tflite("model_generator.keras", "model_generator.tflite")
export_tflite("model_discriminator.keras", "model_discriminator.tflite")