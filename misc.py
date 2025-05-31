from config import *

def get_generator_model_path_at_given_epoch(i):
    return model_path + 'generator_epoch_' + str(i) + ".keras"

def get_discriminator_model_path_at_given_epoch(i):
    return model_path + 'generator_epoch_' + str(i) + ".keras"