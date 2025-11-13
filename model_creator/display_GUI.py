import time

from ganalyzer.GUI import GUI
from ganalyzer.misc import get_all_models, get_available_epochs

# load all the models
t0 = time.time()
available_epochs = get_available_epochs()
generator_list = get_all_models("generator", available_epochs)
discriminators_list = get_all_models("discriminator", available_epochs)
t1 = time.time()
print("==> Time taken to load : ", round(t1 - t0, 2))

print("==> Number of loaded models : ", len(generator_list))

main_gui = GUI(generator_list, discriminators_list)
