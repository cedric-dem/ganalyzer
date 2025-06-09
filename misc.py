from config import *

import keras
import os
import numpy as np


def get_generator_model_path_at_given_epoch(i):
    return get_model_path_at_given_epoch("generator", i)


def get_discriminator_model_path_at_given_epoch(i):
    return get_model_path_at_given_epoch("discriminator", i)


def get_model_path_at_given_epoch(model_type, i):
    return model_path + model_type + "_epoch_" + "0" * (6 - len(str(i))) + str(i) + ".keras"


def get_model_path_at_given_epoch_closest_possible(model_type, i, available_epochs):

    current_best_distance = None
    current_best_result = None

    for available_epoch in available_epochs:
        this_distance = abs(available_epoch - i)
        if current_best_distance is None or current_best_distance > this_distance:
            current_best_distance = this_distance
            current_best_result = available_epoch

    return get_model_path_at_given_epoch(model_type, current_best_result)


def get_available_epochs():
    models_list = get_list_of_keras_models()
    models_list_discriminators = [i for i in models_list if i.startswith("discriminator")]
    epochs_list = []
    for model in models_list_discriminators:
        epochs_list.append(int(model.split("_")[-1].split(".")[0]))
    return epochs_list


def get_all_models(model_type, available_epochs):
    # This function, and get_model_path_at_given_epoch_closest_possible, other than being poorly named, are absurdly not optimal
    # I am ashamed to have written that, but in this case that will be fast enough

    models_quantity = get_current_epoch()

    result = [None for i in range(models_quantity)]

    take_every = models_quantity // load_quantity_gui

    for i in range(models_quantity):
        if i % take_every == 0 or i == 0 or i == models_quantity - 1:
            this_filename = get_model_path_at_given_epoch_closest_possible(model_type, i, available_epochs)
            print("=> will load  " + model_type + " epoch ", i, " closest found is : ", this_filename)
            result[i] = keras.models.load_model(this_filename)

    return result


def find_limits_and_project(arr):
    projected = project_array(arr, 254, np.min(arr), np.max(arr))
    return np.round(projected).astype(np.uint8)


def project_array(arr, to, project_from, project_to):
    delta = project_to - project_from
    if delta > 0:
        result = ((arr - project_from) / delta) * to
    else:
        result = arr
    return result


def get_list_of_keras_models():
    complete_list = os.listdir(model_path)

    complete_list.sort()

    keras_models = [f for f in complete_list if not f.endswith(".csv")]
    return keras_models


def get_current_epoch():
    keras_models = get_list_of_keras_models()

    if len(keras_models) == 0:
        result = 0
    else:
        result = int(keras_models[-1].split("_")[-1].split(".")[0])
    return result
