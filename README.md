# Keras to Tensorflow Tutorial
So you’ve built an awesome machine learning model in Keras and now you want to run it natively through Tensorflow. This tutorial will show you how.

[Keras](http://keras.io/) is a wonderful high level framework for building machine learning models. It is able to utilize multiple backends such as [Tensorflow](http://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/) to do so. When a keras model is saved via the [.save method](http://keras.io/getting-started/faq/#how-can-i-save-a-keras-model), the canonical save method serializes to an HDF5 format. Tensorflow works with [Protocol Buffers](http://developers.google.com/protocol-buffers/), and therefore loads and saves .pb files. This tutorial demonstrates how to:
  * build a basic example using Convolutional Neural Network in Keras for image classification
  * save the Keras model as an HDF5 model
  * verify the Keras model
  * convert the HDF5 model to a Protocol Buffer
  * build a Tensorflow C++ shared library
  * utilize the .pb in a pure Tensorflow app
  * We will utilize Tensorflow's own example code for this

I am conducting this tutorial on macOS and Linux, using Tensorflow version 1.10 and Keras version 2.1.5 .

## Assumptions ##
  * You are familiar with Python *(and C++ if you're interested in the C++ portion of this tutorial)*
  * You are familiar with Keras and Tensorflow and already have your dev environment setup
  * Example code is utilizing Python 3.5, if you are using 2.7 you may have to make modifications

So we suppose you cloned this repository and you accessed it 

> https://github.com/Aazhar/keras2tensorflow && cd keras2tensorflow

## The dataset  ##

In this tutorial we use the flowers classification data from the Tensorflow examples. It’s about 218 MB, it's already included in this repository (downloaded from http://download.tensorflow.org/example_images/flower_photos.tgz).

## Train your model  ##

A simple CNN as an example, however the techniques to port the models work equally well with the built-in Keras models such as Inception and ResNet. I have no illusions that this model will win any awards, but it will serve our purpose.

Few things to note from the code listed below:

* Label your input and output layer(s) – this will make it easier to debug when the model is converted.
* I’m relying on the [Model Checkpoint](https://keras.io/callbacks/#modelcheckpoint) to save my .h5 files – you could also just call classifier.save after the training is complete.
* Make note of the shape parameter you utilize, we will need that when we run the model later.

see [k2tf_trainer.py](https://github.com/Aazhar/keras2tensorflow/blob/master/k2tf_trainer.py) .

No train the model:

> python k2tf_trainer.py --test=./data/flowers/raw-data/validation --train=./data/flowers/raw-data/train --cats=5 --shape=80 --batch=120 --epochs=400 --output=./temp

A few runs of this yielded val_acc in the 83-86% range, and while it’s no Inception, it’s good enough for this exercise.

## Test your model  ##

Let’s just do a quick gut-check on our model – here’s a small script to load your model, image, shape and indices (especially if you didn’t use the flowers set) , see [k2tf_eval.py](https://github.com/Aazhar/keras2tensorflow/blob/master/k2tf_eval.py).

Run the test : 

>python k2tf_eval.py -m '/home/work/keras2tensorflow/temp/k2tf-20181025175144/e-075-vl-0.481-va-0.837.h5' -i './data/flowers/raw-data/validation/dandelion/13920113_f03e867ea7_m.jpg' -s 80

## Convert from HDF5 to .pb  ##

Attribution: This script was adapted from [https://github.com/amir-abdi/keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow).

Adaptation were made from the link above to a script we can run from the command line. The code is almost identical except for the argument parsing. This code does the following:

* Loads your .h5 file
* Replaces your output tensor(s) with a named [Identity Tensor](https://www.tensorflow.org/api_docs/python/tf/identity) – this can be helpful if you are using a model you didn’t build and don’t know all of the output names (of course you could go digging, but this avoids that).
* Saves an ASCII representation of the graph definition. I use this to verify my input and output names for Tensorflow. This can be useful in debugging.
* Replaces all variables within the graph to constants.
* Writes the resulting graph to the output name you specify in the script.

And now let’s run this little guy on our trained model.
>python k2tf_convert.py -m '/home/work/keras2tensorflow/temp/k2tf-20181025175144/e-075-vl-0.481-va-0.837.h5' -n 1


As you can see, two files were written out. An ASCII and .pb file. Have a look at the graph structure, notice the input node name “firstConv2D_input” and the output name “k2tfout_0”, , we will use those in the next section.


## Running your Tensorflow model with Python ##

We need to supply the following arguments to run the Python script :

* the output_graph.pb we generated above
* the labels file – this is supplied with the dataset but you could generate a similar labels.txt from the indices.txt file we produced in our Keras model training
* input width and height. Remember I trained with 80×80 so I must adjust for that here
* The input layer name – I find this in the generated ASCII file from the conversion we did above. In this case it is “firstConv2D_input” – Remember our k2tf_trainer.py named the first layer “firstConv2D”.
* The output layer name – We created this with prefix and can verify it in our ASCII file. We went with the script default which was “k2tfout_0”
* Finally, the image we want to process.

> python label_image.py --graph=./output_graph.pb --labels=./data/flowers/raw-data/labels.txt --input_width=80 --input_height=80 --input_layer=firstConv2D_input --output_layer=k2tfout_0 --image=./data/flowers/raw-data/validation/dandelion/13920113_f03e867ea7_m.jpg

## Running your Tensorflow model with C++ ##

To run our models in C++ we first need to obtain the Tensorflow source tree. The instructions are [here](https://www.tensorflow.org/install/source), but we’ll walk through them below.

Install all dependencies

> git clone https://github.com/tensorflow/tensorflow && cd tensorflow/ && git checkout r1.10 && ls

You can install all dependencies using :

> python tensorflow/tools/pip_package/setup.py install

On Linux or Mac Tensorflow uses [Bazel](https://bazel.build/) as build tool, Windows uses CMake.

After installing bazel :
> ./configure

So here you can just skip all of suggested extensions.

Looking through the bazel BUILD files to find targets you will find the right target, and you can build the C++ extension using :

> bazel build --jobs=6 --verbose_failures //tensorflow:libtensorflow_cc.so

(Up to you to choose the correct number of jobs before going to take a nap while it's compiling)

Once it's done, you should end up with bazel-bin/tensorflow/libtensorflow_cc.so, you need to copy them then, either to common lib directory or current one :

> cd .. & cp -R tensorflow/bazel-bin/tensorflow/libtensorflow_* ./

If you have cmake, there is a CMakeList file that links the .so and all of the required header locations. I’m including it for reference – you DO NOT need cmake to build this. In fact, the g++ commands are provided to build.

> g++ -c -pipe -g -std=gnu++11 -Wall -W -fPIC -I. -I./tensorflow -I./tensorflow/bazel-tensorflow/external/eigen_archive -I./tensorflow/bazel-tensorflow/external/protobuf_archive/src -I./tensorflow/bazel-genfiles -o main.o ./main.cpp

> g++ -o k2tf main.o -ltensorflow_cc -ltensorflow_framework

Now you should have an executable in your directory and we can test our application:

> ./k2tf --graph=./output_graph.pb --labels=./data/flowers/raw-data/labels.txt --input_width=80 --input_height=80 --input_layer=firstConv2D_input --output_layer=k2tfout_0 --image=./data/flowers/raw-data/validation/dandelion/13920113_f03e867ea7_m.jpg

I hope this was useful.

Special attribution to bitbionic.