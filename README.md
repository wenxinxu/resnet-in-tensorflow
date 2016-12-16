# ResNet in Tensorflow
I implement [residual networks](http://arxiv.org/abs/1512.03385) and [its variants](https://arxiv.org/abs/1603.05027) in tensorfflow. 

This implementation is designed to be straightforward and friendly to new ResNet users. You can train a resnet on cifar10 by downloading and running the code. There are screen outputs, tensorboard statistics and tensorboard graph visualization to help you monitor the training process and visualize the model.

####If you like the code, please star it! You are welcome to post questions and suggestions on my github.


##Table of Contents
* [Validation errors](#validation-errors)
* [Training curves](#training-curves)
* [User's guide](#users-guide)
   * [Pre-requisites](#pre-requisites)
   * [Overall structure](#overall-structure)
   * [Hyper-parameters](#hyper-parameters)
   * [Resnet Strcuture](#resnet-structure)
   * [Training](#training)


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

### Pre-requisites
pandas, numpy , opencv, tensorflow(0.11.0, I am not sure if earlier version would work)

### Overall structure
There are four python files in the repository. cifar10_input.py, resnet.py, cifar10_train.py, hyper_parameters.py.

cifar10_input.py includes helper functions to download, extract and pre-process the cifar10 images. 
resnet.py defines the resnet structure.
cifar10_train.py is responsible for the training and validation.
hyper_parameters.py defines hyper-parameters related to train, resnet structure, data augmentation, etc. 

The following parts expain the codes in details.

------------------------------------------------------------------------------------------------------------------------------------
### hyper-parameters
The hyper_parameters.py file defines all the hyper-parameters that you may change to customize your training. All of them are defined via tf.app.flags.FLAGS, so that you may use `python cifar10_train.py --hyper_parameter1=value1 --hyper_parameter2=value2` to set all the hyper-parameters when running. You may also change the default values inside the python script.

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


### ResNet Structure
Here we use the latest version of ResNet. The structure of the residual block looks like [ref](https://arxiv.org/abs/1603.05027):

<img src="https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/appendix/Residual_block.png" width="240">

The inference() function is the main function of resnet.py. It takes three arguments: input_tensor_batch, n and resue. input_tensor_batch is a 4D tensor with shape of [batch_size, img_height, img_width, img_depth]. n is the num_residual_blocks. Reuse is a boolean, indicating the graph is build for train or validation data. 

To enable the different sizes of validation batch to train batch, I use two different sets of placeholders for train and validation data, and build the graphs separately, and the validation graph shares the same weights with the train graph. In this situation, we are passing reuse=True to each variable scope of train graph to fetch the weights. To read more about variable scope, see [variable scope](https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html)


### Training
The class Train() defines all the functions that regulates training, with train() as the main function. The basic idea is to run train_op for FLAGS.train_steps times. If step % FLAGS.report_freq == 0, it will valdiate once, train once and wrote all the summaries onto the tensorboard. (We do want to validate before training, so that we can check the original errors and losses with the theoretical value.)

The following two concepts may help you understand the code better.

####1. Placeholder
Placeholders can be viewed as tensors that must be fed with real data on every execution. If you want to change the "values" of certain tensors on each step of training, placeholders are the most straightforward way. For example, we train the model with different batches of data on each step by feeding different batches of numpy array into the image_placeholder and label_placeholder. A feed dict looks like:
```
feed_dict = {self.image_placeholder: train_batch_data,
             self.label_placeholder: train_batch_labels,
             self.vali_image_placeholder: validation_batch_data,
             self.vali_label_placeholder: validation_batch_labels,
             self.lr_placeholder: FLAGS.init_lr}
```             
For more detailed explaination, see [tf.placeholder()](https://www.tensorflow.org/api_docs/python/io_ops/placeholders#placeholder) and [feeding data](https://www.tensorflow.org/how_tos/reading_data/#feeding)

####2. Summary
Tensorboard is a very useful tool to supervise and visualize the training process. Here I provide a step-by-step guide on how to set up tensorboard.

**a) Summarize the tensors of interest**
After you create the tensor, add `tf.scalar_summary(name='name_on_tensorboard', tensor=tensor)`. [This summary](https://www.tensorflow.org/api_docs/python/summary/generation_of_summaries_#scalar) is essentially an operation. It won't do anything until you run it in a session!

**b) Merge all summaries**
After you set up all the scalar summaries, type `summary_op = tf.merge_all_summaries()`. This command merge all the summarizing operations into a single operation, which means that running summary_op is equivalent to running all the scalar summaries together. 
   




