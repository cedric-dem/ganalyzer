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


def get_all_models(model_type):

    models_quantity = get_current_epoch()

    result = [None for i in range(models_quantity)]

    for i in range(models_quantity):

        this_filename = get_model_path_at_given_epoch(model_type, i)

        if os.path.exists(this_filename):
            print("=> Attempt to load " + model_type + " epoch ", i)
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
