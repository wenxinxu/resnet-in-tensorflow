# ResNet in Tensorflow
Re-implement Kaiming He's deep residual networks with tensorflow (http://arxiv.org/abs/1512.03385, https://arxiv.org/abs/1603.05027). 
This version is designed to be straightforward and friendly to new ResNet users. You can train a resnet on cifar10 by downloading and running the code directly. There are screen outputs, tensorboard statistics and tensorboard graph visualization to help you understand the model.

####If you like the code, please star it! You are welcome to post questions and suggestions on my github.

## Validation errors
The lowest valdiation errors of ResNet-32, ResNet-56 and ResNet-110 are 6.7%, 6.5% and 6.2% respectively. You can change the number of the total layers by changing the hyper-parameter num_residual_blocks. Total layers = 6 * num_residual_blocks + 2

Network | Lowest Validation Error
------- | -----------------------
ResNet-32 | 6.7%
ResNet-56 | 6.5%
ResNet-110 | 6.2%

## Training curves
![alt tag](https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/train_curve2.png)

## User's guide
Basically, you can run cifar10_train.py and see how it works from the screen output without any downloads. It’s better to define a specific version identifier before running, as the training logs, checkpoints, and error.csv file will be saved in a new logs_$version folder. You may do this by command line commands like: `python cifar10_train.py --version='test'`. You may also change the version number inside the hyper_parameters.py file

The values and statistics of each layer can be found on tensorboard. Use `tensorboard --logdir='logs_$version'` command to pull them out. (For e.g. If the version is ‘test’, the logdir should be ‘logs_test’.)

###	Pre-requisites
pandas, numpy , opencv, tensorflow(0.11.0, I am not sure if earlier version would work)

### Overall structure
There are four python files in the repository. cifar10_input.py, resnet.py, cifar10_train.py, hyper_parameters.py.

cifar10_input.py includes helper functions to download, extract and pre-process the cifar10 images. 
resnet.py defines the resnet structure.
cifar10_train.py is responsible for the training and validation.
hyper_parameters.py defines hyper-parameters related to train, resnet structure, data augmentation, etc. 

The following parts expain the codes in details.

------------------------------------------------------------------------------------------------------------------------------------
### hyper_parameters.py
This file defines all the hyper-parameters that you may change to customize your training. All of them are defined via tf.app.flags.FLAGS, so that you may use `python cifar10_train.py --hyper_parameter1=value1 --hyper_parameter2=value2` to set all the hyper-parameters when running. You may also change the default values inside the python script.

There are five categories of hyper-parameters.

-------------------------------------------------------------------------------------------------------------------------------------
####1. Hyper-parameters about saving training logs, tensorboard outputs and screen outputs, which includes:
**version**: str. The checkpoints and output events will be saved in logs_$version/

**report_freq**: int. How many batches to run a full validation and print screen output once. Screen output looks like:
![alt tag](https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/appendix/Screen_output_example.png)

**train_ema_decay**: float. The tensorboard will record a moving average of batch train errors, besides the original ones. This decay factor is used to define an ExponentialMovingAverage object in tensorflow with `tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)`. Essentially, the recorded error = train_ema_decay * shadowed_error + (1 - train_ema_decay) * current_batch_error. The larger the train_ema_decay is, the smoother the training curve will be.

-------------------------------------------------------------------------------------------------------------------------------------


####2. Hyper-parameters that regulates the training process
**train_steps**: int. Total steps you want to train

**is_full_validation**: boolean. If you want to use all the 10000 validation images to run the validation (True), or you want to randomly draw a batch of validation data (False)

**train_batch_size**: int. Train batch size...

**validation_batch_size**: int. Validation batch size...

**init_lr**: float. The initial learning rate when started. The learning rate may decay based on your setting

**lr_decay_factor**: float. On each decay, the learning rate will become lr_decay_factor * current_learning_rate

**decay_step0**: int. Which step to decay the learning rate on for the first time

**decay_step1**: int. Which step to decay the learning rate on for the second time

------------------------------------------------------------------------------------------------------------------------------------

####3. Hyper-parameters that modifies the network
**num_residual_blocks**: int. The total layers of the ResNet = 6 * num_residual_blocks + 2

**weight_decay**: float. weight decay used to regularize the network. Total_loss = train_loss + weight_decay*sum(weights)

-----------------------------------------------------------------------------------------------------------------------------------

####4. About data augmentation
**padding_size**: int. Padding and random cropping during training can prevent overfitting. padding_size is numbers of zero pads to add on each side of the image.

-----------------------------------------------------------------------------------------------------------------------------------

####5. Loading checkpoints
**ckpt_path**: str. The path of the checkpoint that you want to load

**is_use_ckpt**: Whether to load a checkpoint and continue training?

-----------------------------------------------------------------------------------------------------------------------------------



