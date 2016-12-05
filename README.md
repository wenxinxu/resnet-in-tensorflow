# resnet_in_tensorflow
Re-implement Kaiming He's deep residual networks with tensorflow (http://arxiv.org/abs/1512.03385, https://arxiv.org/abs/1603.05027). 
This version is designed to be straightforward and friendly to new ResNet users. You can train a resnet on cifar10 by downloading and running the code directly. There are screen outputs, tensorboard statistics and tensorboard graph visualization to help you understand the model.

####If you like the code, please star it! You are welcome to post questions and suggestions on my github.

## Validation errors
I implemented 32, 56 and 110 layers of resnet respectively. The lowest valdiation errors are 6.7%, 6.6% and 6.4%.

## Training curves
![alt tag](https://github.com/wenxinxu/resnet_in_tensorflow/blob/master/train_curve2.png)

## User's guide
There are four python files in the repository. cifar10_train.py, hyper_parameters.py, cifar10_input.py and resnet.py. 

Basically you can run cifar10_train.py and see how it works from the screen output without any downloads. It’s better to define a specific version identifier before running, as the training logs, checkpoints and error.csv file will be saved in a new logs_$version folder. You may do this by command line orders like python cifar10_train.py --version=’test’ or change inside the hyper-parameter.py. 

The values and statistics of each layer can be found on tensorboard. Use tensorboard --logdir=’logs_$version’ command to see them. (For eg. If the version is ‘test’, the logdir should be ‘logs_test’.)

###	pre-requisites
pandas, numpy , opencv, tensorflow(0.11.0, I am not sure if earlier version would work)
###	hyper-parameters.py
Defines hyper-parameters related to train, resnet structure, data augmentation, etc. 
### resnet.py
The resnet structure. 
###	cifar10_train.py
Run this code and start training immediately!


