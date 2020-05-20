# NumPyCNN: Implementing Convolutional Neural Networks From Scratch
NumPyCNN is a Python implementation for convolutional neural networks (CNNs) from scratch using NumPy. 

**IMPORTANT** *If you are coming for the code of the tutorial titled [Building Convolutional Neural Network using NumPy from Scratch]( [https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad/)), then it has been moved to the [TutorialProject](https://github.com/ahmedfgad/NumPyCNN/tree/master/TutorialProject) directory on 20 May 2020.*

The project has a single module named `numpycnn.py` which implements all classes and functions needed to build the CNN.

It is very important to note that the project only implements the **forward pass** of training CNNs and there is **no learning algorithm used**. Just the learning rate is used to make some changes to the weights after each epoch which is better than leaving the weights unchanged.

The project can be used for classification problems where only 1 class per sample is allowed.

The project will be extended to **train CNN using the genetic algorithm** with the help of a library named [PyGAD](https://pypi.org/project/pygad). Check the library's documentation at [Read The Docs](https://pygad.readthedocs.io/): https://pygad.readthedocs.io

Install it via pip:

```python
pip install pygad
```

# Supported Layers

The project implements the following layers:

1. Convolution
2. Dense
3. Average Pooling
4. Max Pooling
5. ReLU
6. Flatten

# Supported Activation Functions

It supports the following activation functions:

1. ReLU
2. Sigmoid
3. Softmax

# Steps to Use the Project

1.	Prepare the training data.
2.	Build network architecture.
3.	Network architecture summary.
4.	Train the network.
5.	Make predictions.

## Prepare the Training Data

The training data is prepared as 2 arrays, one for the inputs and another for the outputs. 

The input array is a 4-dimensional array with the following dimensions:

1. Number of training samples.
2. Width of the sample.
3. Height of the sample.
4. Number of channels in the sample. Set to 1 if the input has no channels.

The output array has only 1 value per sample representing the class label.

Attached to the project 2 NumPy arrays created out of 4 classes from the Fruits360 image dataset. The classes names are **Apple Braeburn**, **Lemon Meyer**, **Mango**, and **Raspberry**.

The images and their class labels are saved as 2 `.npy` files named:

1. dataset_inputs.npy

2. dataset_outputs.npy

For the purpose of just demonstrating how thins are working, only 20 samples per class are used and thus there is a total of 80 samples in the dataset. 

The dataset_inputs.npy file holds the dataset inputs and dataset_outputs.npy holds the outputs. The shape of the inputs is `(80, 100, 100, 3)` where the single image shape is `(100, 100, 3)`. The shape of the outputs is `(80)`.

Here is how the 2 files are read:

```python
train_inputs = numpy.load("dataset_inputs.npy")
train_outputs = numpy.load("dataset_outputs.npy")
```



## Build Network Architecture

A network of the following architecture is built.

- Input
- Conv: With 2 filters.
- ReLU
- Average Pooling
- Conv: With 3 filters.
- ReLU
- Max Pooling
- Flatten
- Dense
- Dense: With 4 output neurons because the data has 4 classes.

Here is the code for building such a network. Remember to set the `num_classes` variable according to the number of classes in the dataset.

```python
sample_shape = train_inputs.shape[1:]

input_layer = numpycnn.Input2D(input_shape=sample_shape)
conv_layer1 = numpycnn.Conv2D(num_filters=2,
                              kernel_size=3,
                              previous_layer=input_layer,
                              activation_function="relu")
relu_layer1 = numpycnn.ReLU(previous_layer=conv_layer1)
average_pooling_layer = numpycnn.AveragePooling2D(pool_size=2, 
                                                  previous_layer=relu_layer1,
                                                  stride=2)

conv_layer2 = numpycnn.Conv2D(num_filters=3,
                              kernel_size=3,
                              previous_layer=average_pooling_layer,
                              activation_function=None)
relu_layer2 = numpycnn.ReLU(previous_layer=conv_layer2)
max_pooling_layer = numpycnn.AveragePooling2D(pool_size=2, 
                                              previous_layer=relu_layer2,
                                              stride=2)

conv_layer3 = numpycnn.Conv2D(num_filters=1,
                              kernel_size=3,
                              previous_layer=max_pooling_layer,
                              activation_function=None)
relu_layer3 = numpycnn.ReLU(previous_layer=conv_layer3)
pooling_layer = numpycnn.AveragePooling2D(pool_size=2, 
                                          previous_layer=relu_layer3,
                                          stride=2)

flatten_layer = numpycnn.Flatten(previous_layer=pooling_layer)
dense_layer1 = numpycnn.Dense(num_neurons=100, 
                              previous_layer=flatten_layer,
                              activation_function="relu")
num_classes = 4
dense_layer2 = numpycnn.Dense(num_neurons=num_classes, 
                              previous_layer=dense_layer1,
                              activation_function="softmax")
```

## Network Architecture Summary

The `summary()` function prints a summary of the network architecture.

```python
numpycnn.summary(last_layer=dense_layer2)
```

```python
----------Network Architecture----------
<class 'numpycnn.Input2D'>
<class 'numpycnn.Conv2D'>
<class 'numpycnn.ReLU'>
<class 'numpycnn.AveragePooling2D'>
<class 'numpycnn.Conv2D'>
<class 'numpycnn.ReLU'>
<class 'numpycnn.MaxPooling2D'>
<class 'numpycnn.Flatten'>
<class 'numpycnn.Dense'>
<class 'numpycnn.Dense'>
----------------------------------------
```

Training the Network

The `train()` function trains the network. It accepts a parameter named `last_layer` which refers to the output layer in the network. Besides the training data inputs and outputs, it accepts the number of epochs and the learning rate.

```python
numpycnn.train(last_layer=dense_layer2, 
               train_inputs=train_inputs, 
               train_outputs=train_outputs, 
               epochs=2,
               learning_rate=0.1)
```

## Making Predictions

After the network is trained, the `predict()` function can be used for making predictions.

```python
predictions = numpycnn.predict(last_layer=dense_layer2, train_inputs=train_inputs)
```

# Results Visualization

The first **conv-relu-pool** layers:
![l1](https://user-images.githubusercontent.com/16560492/39051349-ac56ac56-44a8-11e8-8695-29901dd3a811.png)

The second **conv-relu-pool** layers:
![l2](https://user-images.githubusercontent.com/16560492/39051582-6abe0996-44a9-11e8-88e1-589a673a8b11.png)

The last **conv-relu-pool** layers:
![l3](https://user-images.githubusercontent.com/16560492/39051603-76339f3e-44a9-11e8-8e4e-9303a51aaa79.png)

# For More Information

There are different resources that can be used to get started with the building CNN and its Python implementation. 

## Tutorial: Building CNN in Python

To start with coding the genetic algorithm, you can check the tutorial titled [**Building Convolutional Neural Network using NumPy from Scratch**](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a)
- [KDnuggets](https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html)
- [Chinese Translation](http://m.aliyun.com/yunqi/articles/585741)

[This tutorial](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) is prepared based on a previous version of the project but it still a good resource to start with coding the genetic algorithm.

[![Building CNN in Python](https://user-images.githubusercontent.com/16560492/82431022-6c3a1200-9a8e-11ea-8f1b-b055196d76e3.png)](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)

## Tutorial: Derivation of CNN from FCNN

Get started with the genetic algorithm by reading the tutorial titled [**Derivation of Convolutional Neural Network from Fully Connected Network Step-By-Step**](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad) which is available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad)
* [Towards Data Science](https://towardsdatascience.com/derivation-of-convolutional-neural-network-from-fully-connected-network-step-by-step-b42ebafa5275)
* [KDnuggets](https://www.kdnuggets.com/2018/04/derivation-convolutional-neural-network-fully-connected-step-by-step.html)

[![Derivation of CNN from FCNN](https://user-images.githubusercontent.com/16560492/82431369-db176b00-9a8e-11ea-99bd-e845192873fc.png)](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad)

## Book: Practical Computer Vision Applications Using Deep Learning with CNNs

You can also check my book cited as [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665) which discusses neural networks, convolutional neural networks, deep learning, genetic algorithm, and more.

Find the book at these links:

- [Amazon](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665)
- [Springer](https://link.springer.com/book/10.1007/978-1-4842-4167-7)
- [Apress](https://www.apress.com/gp/book/9781484241660)
- [O'Reilly](https://www.oreilly.com/library/view/practical-computer-vision/9781484241677)
- [Google Books](https://books.google.com.eg/books?id=xLd9DwAAQBAJ)

![Fig04](https://user-images.githubusercontent.com/16560492/78830077-ae7c2800-79e7-11ea-980b-53b6bd879eeb.jpg)

# Contact Us

* E-mail: ahmed.f.gad@gmail.com
* [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
* [Amazon Author Page](https://amazon.com/author/ahmedgad)
* [Heartbeat](https://heartbeat.fritz.ai/@ahmedfgad)
* [Paperspace](https://blog.paperspace.com/author/ahmed)
* [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
* [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
* [GitHub](https://github.com/ahmedfgad)
