import tensorflow as tf
import numpy as np


def display_intermediary_layers(model_path, input_data):

    print("======================> Now on model : ", model_path)

    model = tf.keras.models.load_model(model_path)
    for i in range(len(model.layers)):
        layer_output = tf.keras.Model(inputs=model.inputs, outputs=model.layers[i].output).predict(input_data)

        print("===> layer ", i," name : ",model.layers[i].name, " shape ", layer_output.shape, " min value", np.min(layer_output), " max ", np.max(layer_output))


discriminator_path = "model_test_1/discriminator_epoch_53.keras"
generator_path = "model_test_1/generator_epoch_53.keras"

input_image = np.random.randint(0, 254, size=(1, 64, 64, 3), dtype=np.uint8)
latent_vector_noise = np.random.normal(loc=0, scale=1, size=(1, 169)).astype(np.float32)

display_intermediary_layers(discriminator_path, input_image)
display_intermediary_layers(generator_path, latent_vector_noise)
