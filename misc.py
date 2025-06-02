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
    while os.path.isfile(this_filename) and i<4:
        result.append(keras.models.load_model(this_filename))
        this_filename=get_model_path_at_given_epoch(model_type,i)
        print("=> Attempt to load "+model_type+" epoch ",i)
        i+=1
    return result

def find_limits_and_project(arr):
    projected=project_array(arr, 254, np.min(arr), np.max(arr))
    return np.round(projected).astype(np.uint8)

def project_array(arr, to, project_from, project_to):
    delta=project_to-project_from
    if delta>0:
        result = ((arr-project_from) / delta) * to
    else:
        result = arr
    return result
