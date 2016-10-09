# resnet_in_tensorflow
Re-implement Kaiming He's deep residual networks with tensorflow (http://arxiv.org/abs/1512.03385, https://arxiv.org/abs/1603.05027). You can train a resnet on cifar10 by downloading and running the code directly.

## The repository is consist of three python scripts. 
cifar10_train.py is the main body the code. It will download the data set and start training and validation directly. 
The version flag helps manage the version of your experiments. The logs will be saved in 'logs_version' folder and all training curves/scalar/histogram summary can be checked by tensorboard. 
The num_residual_blocks flag defines the layer number of the network. Total layer = 6 * num_residual_blocks + 3

resnet.py defines the network structure and some summary operations to summarize the sparsity of each layer.You can adjust the weight decay by changing the 'weight_decay' and 'fc_weight_decay'

cifar10_input.py is the data I-O file. Supports training on random label. 

## TODO:
1. Put a training curve on
2. Add a bottleneck structure for networks with more than 56 layers
