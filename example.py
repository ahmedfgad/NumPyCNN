import numpy
import numpycnn

"""
Convolutional neural network implementation using NumPy
A tutorial that helps to get started (Building Convolutional Neural Network using NumPy from Scratch) available in these links: 
    https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad
    https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
    https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741
"""

train_inputs = numpy.load("dataset_inputs.npy")
train_outputs = numpy.load("dataset_outputs.npy")

sample_shape = train_inputs.shape[1:]
num_classes = 4

input_layer = numpycnn.Input2D(input_shape=sample_shape)
conv_layer1 = numpycnn.Conv2D(num_filters=2,
                              kernel_size=3,
                              previous_layer=input_layer,
                              activation_function=None)
relu_layer1 = numpycnn.ReLU(previous_layer=conv_layer1)
average_pooling_layer = numpycnn.AveragePooling2D(pool_size=2, 
                                                  previous_layer=relu_layer1,
                                                  stride=2)

conv_layer2 = numpycnn.Conv2D(num_filters=3,
                              kernel_size=3,
                              previous_layer=average_pooling_layer,
                              activation_function=None)
relu_layer2 = numpycnn.ReLU(previous_layer=conv_layer2)
max_pooling_layer = numpycnn.MaxPooling2D(pool_size=2, 
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
dense_layer2 = numpycnn.Dense(num_neurons=num_classes, 
                              previous_layer=dense_layer1,
                              activation_function="softmax")

numpycnn.summary(last_layer=dense_layer2)

numpycnn.train(last_layer=dense_layer2, 
               train_inputs=train_inputs, 
               train_outputs=train_outputs, 
               epochs=1,
               learning_rate=0.1)

predictions = numpycnn.predict(last_layer=dense_layer2, train_inputs=train_inputs)
print(predictions)
