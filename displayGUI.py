import time

from GUI import *

# load all the models
t0 = time.time()
generator_list = get_all_models("generator")
discriminators_list = get_all_models("discriminator")
t1 = time.time()
print("==> Time taken to load : ", round(t1 - t0, 2))

print("==> Number of loaded models : ", len(generator_list))

main_gui = GUI(generator_list, discriminators_list)
