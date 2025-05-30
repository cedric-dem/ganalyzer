# ganalyzer

## GUI to experiment with GAN

## train.py
The train.py file launch the training procedure, save the discriminator and generator models at every epochs in .keras file, in the model folder. 
It continue the training process based on the latest models in the model folders
also adds lots of statistics about the training process in a csv file

## displayGUI.py
play around with the GUI and impact, in real time, both models
(will) contain demo model to avoid training 

## displayStats.py
plot content of .csv statistics saved during the train process