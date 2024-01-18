In this repo, I have implemented from scratch, using Pytorch, a Transformer decoder architecture, with the aim of generate new text, coherent with the text used in training.
The result are shown in the DiVinoCommedia(output).txt file, emulating the poem "La Divina Commedia" by Dante Alighieri, which was the text given as training set.
It has been done in my free time and does not involve any educational course I have taken. 
For this reason, it is an incomplete and not super coherent model, I'll work on it as soon as possible.
The architecture is inspired by the Karpanthy's model NanoGPT, clearly explaind on his YouTube channel.

The name used are inspired by Transformers movies, where BumbleBee was the talking impaired human/machine. 
The model is thus fully contained in BumbleBee.py, whereas Radio.py is a file used to load the pre-trained dictionary by BumbleBee and generate text, decoupling training
and generation steps.
