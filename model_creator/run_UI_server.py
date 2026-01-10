
from config import GUI_tkinter
from ganalyzer.GUITkinter import GUITkinter
from ganalyzer.GUIWebPage import GUIWebPage, get_models_generator_and_discriminator

if GUI_tkinter:
	generators_list, discriminators_list = get_models_generator_and_discriminator("model_0_small", 121)

	main_gui = GUITkinter(generators_list, discriminators_list)

else:
	main_gui = GUIWebPage()
