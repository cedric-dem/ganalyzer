from config import *

import keras
import os
import numpy as np

def get_generator_model_path_at_given_epoch(i):
    return get_model_path_at_given_epoch("generator", i)

def get_discriminator_model_path_at_given_epoch(i):
    return get_model_path_at_given_epoch("discriminator", i)

def get_model_path_at_given_epoch(model_type, i):
    return model_path + model_type +'_epoch_' + str(i) + ".keras"

def get_all_models(model_type):
    result=[]
    i=0
    this_filename=get_model_path_at_given_epoch(model_type,0)
    while os.path.isfile(this_filename) and i<10:
        result.append(keras.models.load_model(this_filename))
        this_filename=get_model_path_at_given_epoch(model_type,i)
        print("=> Attempt to load epoch ",i)
        i+=1
    return result

def set_interval(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    delta = max_val - min_val
    if delta>0:
        projected = (arr - min_val) / delta * 254
    else:
        projected = arr

    return np.round(projected).astype(np.uint8)

def project_array(arr, to, project_from, project_to):
    return ((arr-project_from)/(project_to-project_from))*to