import numpy
import functools

"""
Convolutional neural network implementation using NumPy
A tutorial that helps to get started (Building Convolutional Neural Network using NumPy from Scratch) available in these links: 
    https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad
    https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
    https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741
"""

def sigmoid(sop):

    """
    Applies the sigmoid function.

    sop: The input to which the sigmoid function is applied.

    Returns the result of the sigmoid function.
    """

    if type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    return 1.0 / (1 + numpy.exp(-1 * sop))

def relu(sop):

    """
    Applies the rectified linear unit (ReLU) function.

    sop: The input to which the relu function is applied.

    Returns the result of the ReLU function.
    """

    if not (type(sop) in [list, tuple, numpy.ndarray]):
        if sop < 0:
            return 0
        else:
            return sop
    elif type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    result = sop
    result[sop < 0] = 0

    return result

def softmax(layer_outputs):

    """
    Applies the sotmax function.

    sop: The input to which the softmax function is applied.

    Returns the result of the softmax function.
    """
    return layer_outputs / (numpy.sum(layer_outputs) + 0.000001)

supported_activation_functions = ("sigmoid", "relu", "softmax")

class Input2D:

    """
    Implementing the input layer of a CNN.
    """

    def __init__(self, input_shape):

        if len(input_shape) < 2:
            raise ValueError("The Input2D class creates an input layer for data inputs with at least 2 dimensions but ({num_dim}) dimensions found.".format(num_dim=len(input_shape)))
        elif len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1], 1)

        for dim_idx, dim in enumerate(input_shape):
            if dim <= 0:
                raise ValueError("The dimension size of the inputs cannot be <=0. Please pass a valid value to the 'input_size' parameter.")
            # The number of neurons in the input layer.
        self.input_shape = input_shape
        self.layer_output_size = input_shape

class Conv2D:

    def __init__(self, num_filters, kernel_size, previous_layer, activation_function=None):

        if num_filters <= 0:
            raise ValueError("Number of filters cannot be <= 0. Please pass a valid value to the 'num_filters' parameter.")
        # Number of filters in the conv layer.
        self.num_filters = num_filters

        if kernel_size <= 0:
            raise ValueError("The kernel size cannot be <= 0. Please pass a valid value to the 'kernel_size' parameter.")
        # Kernel size of each filter.
        self.kernel_size = kernel_size

        # Validating the activation function
        if (activation_function is None):
            self.activation = None
        elif (activation_function == "relu"):
            self.activation = relu
        elif (activation_function == "sigmoid"):
            self.activation = sigmoid
        elif (activation_function == "softmax"):
            raise ValueError("The softmax activation function cannot be used in a conv layer.")
        else:
            raise ValueError("The specified activation function '{activation_function}' is not among the supported activation functions {supported_activation_functions}. Please use one of the supported functions.".format(activation_function=activation_function, supported_activation_functions=supported_activation_functions))

        self.activation_function = activation_function

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        self.filter_bank_size = (self.num_filters,
                                 self.kernel_size, 
                                 self.kernel_size, 
                                 self.previous_layer.layer_output_size[-1])

        # Initializing the filters of the conv layer.
        self.initial_weights = numpy.random.uniform(low=-0.1,
                                                    high=0.1,
                                                    size=self.filter_bank_size)

        # The trained filters of the conv layer. Only assigned a value after the network is trained (i.e. the train_network() function completes).
        # Just initialized to be equal to the initial filters
        self.trained_weights = self.initial_weights.copy()

        self.layer_input_size = self.previous_layer.layer_output_size

        # Later, it must conider strides and paddings
        self.layer_output_size = (self.previous_layer.layer_output_size[0] - self.kernel_size + 1, 
                                  self.previous_layer.layer_output_size[1] - self.kernel_size + 1, 
                                  num_filters)
        self.layer_output = None

    def conv_(self, input2D, conv_filter):

        result = numpy.zeros((input2D.shape))
        # Looping through the image to apply the convolution operation.
        for r in numpy.uint16(numpy.arange(self.filter_bank_size[1]/2.0, 
                              input2D.shape[0]-self.filter_bank_size[1]/2.0+1)):
            for c in numpy.uint16(numpy.arange(self.filter_bank_size[1]/2.0, 
                                               input2D.shape[1]-self.filter_bank_size[1]/2.0+1)):
                """
                Getting the current region to get multiplied with the filter.
                How to loop through the image and get the region based on 
                the image and filer sizes is the most tricky part of convolution.
                """
                curr_region = input2D[r-numpy.uint16(numpy.floor(self.filter_bank_size[1]/2.0)):r+numpy.uint16(numpy.ceil(self.filter_bank_size[1]/2.0)), 
                                      c-numpy.uint16(numpy.floor(self.filter_bank_size[1]/2.0)):c+numpy.uint16(numpy.ceil(self.filter_bank_size[1]/2.0))]
                # Element-wise multipliplication between the current region and the filter.
                curr_result = curr_region * conv_filter
                conv_sum = numpy.sum(curr_result) # Summing the result of multiplication.

                if self.activation is None:
                    result[r, c] = conv_sum # Saving the SOP in the convolution layer feature map.
                else:
                    result[r, c] = self.activation(conv_sum) # Saving the activation function result in the convolution layer feature map.

        # Clipping the outliers of the result matrix.
        final_result = result[numpy.uint16(self.filter_bank_size[1]/2.0):result.shape[0]-numpy.uint16(self.filter_bank_size[1]/2.0), 
                              numpy.uint16(self.filter_bank_size[1]/2.0):result.shape[1]-numpy.uint16(self.filter_bank_size[1]/2.0)]
        return final_result

    def conv(self, input2D):
        if len(input2D.shape) != len(self.initial_weights.shape) - 1: # Check if there is a match in the number of dimensions between the image and the filters.
            raise ValueError("Number of dimensions in the conv filter and the input do not match.")  
        if len(input2D.shape) > 2 or len(self.initial_weights.shape) > 3: # Check if number of image channels matches the filter depth.
            if input2D.shape[-1] != self.initial_weights.shape[-1]:
                raise ValueError("Number of channels in both the input and the filter must match.")
        if self.initial_weights.shape[1] != self.initial_weights.shape[2]: # Check if filter dimensions are equal.
            raise ValueError('A filter must be a square matrix. I.e. number of rows and columns must match.')
        if self.initial_weights.shape[1]%2==0: # Check if filter diemnsions are odd.
            raise ValueError('A filter must have an odd size. I.e. number of rows and columns must be odd.')

        # An empty feature map to hold the output of convolving the filter(s) with the image.
        feature_maps = numpy.zeros((input2D.shape[0]-self.initial_weights.shape[1]+1, 
                                    input2D.shape[1]-self.initial_weights.shape[1]+1, 
                                    self.initial_weights.shape[0]))

        # Convolving the image by the filter(s).
        for filter_num in range(self.initial_weights.shape[0]):
            # print("Filter ", filter_num + 1)
            curr_filter = self.trained_weights[filter_num, :] # getting a filter from the bank.
            """ 
            Checking if there are mutliple channels for the single filter.
            If so, then each channel will convolve the image.
            The result of all convolutions are summed to return a single feature map.
            """
            if len(curr_filter.shape) > 2:
                conv_map = self.conv_(input2D[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
                for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                    conv_map = conv_map + self.conv_(input2D[:, :, ch_num], 
                                      curr_filter[:, :, ch_num])
            else: # There is just a single channel in the filter.
                conv_map = self.conv_(input2D, curr_filter)
            feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.

        self.layer_output = feature_maps

class MaxPooling2D:

    def __init__(self, pool_size, previous_layer, stride=2):
        if not (type(pool_size) is int):
            raise ValueError("The expected type of the pool_size is int but {pool_size_type} found.".format(pool_size_type=type(pool_size)))

        if pool_size <= 0:
            raise ValueError("The passed value to the pool_size parameter cannot be <= 0.")
        self.pool_size = pool_size

        if stride <= 0:
            raise ValueError("The passed value to the stride parameter cannot be <= 0.")
        self.stride = stride

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        self.layer_input_size = self.previous_layer.layer_output_size

        self.layer_output_size = (numpy.uint16((self.previous_layer.layer_output_size[0] - self.pool_size + 1)/stride + 1), 
                                  numpy.uint16((self.previous_layer.layer_output_size[1] - self.pool_size + 1)/stride + 1), 
                                  self.previous_layer.layer_output_size[-1])
        self.layer_output = None

    def max_pooling(self, input2D):
        
        # Preparing the output of the pooling operation.
        pool_out = numpy.zeros((numpy.uint16((input2D.shape[0]-self.pool_size+1)/self.stride+1),
                                numpy.uint16((input2D.shape[1]-self.pool_size+1)/self.stride+1),
                                input2D.shape[-1]))
        for map_num in range(input2D.shape[-1]):
            r2 = 0
            for r in numpy.arange(0,input2D.shape[0]-self.pool_size+1, self.stride):
                c2 = 0
                for c in numpy.arange(0, input2D.shape[1]-self.pool_size+1, self.stride):
                    pool_out[r2, c2, map_num] = numpy.max([input2D[r:r+self.pool_size,  c:c+self.pool_size, map_num]])
                    c2 = c2 + 1
                r2 = r2 +1

        self.layer_output = pool_out

class AveragePooling2D:

    def __init__(self, pool_size, previous_layer, stride=2):
        if not (type(pool_size) is int):
            raise ValueError("The expected type of the pool_size is int but {pool_size_type} found.".format(pool_size_type=type(pool_size)))

        if pool_size <= 0:
            raise ValueError("The passed value to the pool_size parameter cannot be <= 0.")
        self.pool_size = pool_size

        if stride <= 0:
            raise ValueError("The passed value to the stride parameter cannot be <= 0.")
        self.stride = stride

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        self.layer_input_size = self.previous_layer.layer_output_size

        self.layer_output_size = (numpy.uint16((self.previous_layer.layer_output_size[0] - self.pool_size + 1)/stride + 1), 
                                  numpy.uint16((self.previous_layer.layer_output_size[1] - self.pool_size + 1)/stride + 1), 
                                  self.previous_layer.layer_output_size[-1])
        self.layer_output = None

    def average_pooling(self, input2D):
        
        # Preparing the output of the pooling operation.
        pool_out = numpy.zeros((numpy.uint16((input2D.shape[0]-self.pool_size+1)/self.stride+1),
                                numpy.uint16((input2D.shape[1]-self.pool_size+1)/self.stride+1),
                                input2D.shape[-1]))
        for map_num in range(input2D.shape[-1]):
            r2 = 0
            for r in numpy.arange(0,input2D.shape[0]-self.pool_size+1, self.stride):
                c2 = 0
                for c in numpy.arange(0, input2D.shape[1]-self.pool_size+1, self.stride):
                    pool_out[r2, c2, map_num] = numpy.mean([input2D[r:r+self.pool_size,  c:c+self.pool_size, map_num]])
                    c2 = c2 + 1
                r2 = r2 +1

        self.layer_output = pool_out

class Flatten:

    def __init__(self, previous_layer):

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        self.layer_input_size = self.previous_layer.layer_output_size

        self.layer_output_size = functools.reduce(lambda x, y: x*y, self.previous_layer.layer_output_size)
        self.layer_output = None

    def flatten(self, input2D):

        self.layer_output_size = input2D.size
        self.layer_output = numpy.ravel(input2D)

class ReLU:

    def __init__(self, previous_layer):

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        self.layer_input_size = self.previous_layer.layer_output_size

        self.layer_output_size = self.previous_layer.layer_output_size
        self.layer_output = None

    def relu_layer(self, layer_input):

        self.layer_output_size = layer_input.size
        self.layer_output = relu(layer_input)

class Dense:

    """
    Implementing the input dense (fully connected) layer of a neural network.
    """

    def __init__(self, num_neurons, previous_layer, activation_function="relu"):
        if num_neurons <= 0:
            raise ValueError("Number of neurons cannot be <= 0. Please pass a valid value to the 'num_neurons' parameter.")
        # Number of neurons in the dense layer.
        self.num_neurons = num_neurons

        # Validating the activation function
        if (activation_function == "relu"):
            self.activation = relu
        elif (activation_function == "sigmoid"):
            self.activation = sigmoid
        elif (activation_function == "softmax"):
            self.activation = softmax
        else:
            raise ValueError("The specified activation function '{activation_function}' is not among the supported activation functions {supported_activation_functions}. Please use one of the supported functions.".format(activation_function=activation_function, supported_activation_functions=supported_activation_functions))

        self.activation_function = activation_function

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer
        
        if type(self.previous_layer.layer_output_size) in [list, tuple, numpy.ndarray] and len(self.previous_layer.layer_output_size) > 1:
            raise ValueError("The input to the dense layer must be of type int but {sh} found.".format(sh=type(self.previous_layer.layer_output_size)))
        # Initializing the weights of the layer.
        self.initial_weights = numpy.random.uniform(low=-0.1,
                                                    high=0.1,
                                                    size=(self.previous_layer.layer_output_size, self.num_neurons))

        # The trained weights of the layer. Only assigned a value after the network is trained (i.e. the train_network() function completes).
        # Just initialized to be equal to the initial weights
        self.trained_weights = self.initial_weights.copy()

        self.layer_input_size = self.previous_layer.layer_output_size

        self.layer_output_size = num_neurons
        self.layer_output = None

    def dense_layer(self, layer_input):

        if self.trained_weights is None:
            raise TypeError("The weights of the dense layer cannot be of Type 'None'.")

        sop = numpy.matmul(layer_input, self.trained_weights)

        self.layer_output = self.activation(sop)

class Model:
    
    def __init__(self, last_layer, epochs=10, learning_rate=0.01):
        self.last_layer = last_layer
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.network_layers = self.get_layers()

    def get_layers(self):
        network_layers = []

        layer = self.last_layer

        while "previous_layer" in layer.__init__.__code__.co_varnames:
            network_layers.insert(0, layer)
            layer = layer.previous_layer
        
        return network_layers

    def train(self, train_inputs, train_outputs):
        if (train_inputs.ndim != 4):
            raise ValueError("The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(num_dims=train_inputs.ndim))    

        if (train_inputs.shape[0] != len(train_outputs)):
            raise ValueError("Mismatch between the number of input samples and number of labels: {num_samples_inputs} != {num_samples_outputs}.".format(num_samples_inputs=train_inputs.shape[0], num_samples_outputs=len(train_outputs)))

        network_predictions = []
        network_error = 0
    
        for epoch in range(self.epochs):
            print("Epoch {epoch}".format(epoch=epoch))
            for sample_idx in range(train_inputs.shape[0]):
                # print("Sample {sample_idx}".format(sample_idx=sample_idx))
                self.feed_sample(train_inputs[sample_idx, :])
    
                try:
                    predicted_label = numpy.where(numpy.max(self.last_layer.layer_output) == self.last_layer.layer_output)[0][0]
                except IndexError:
                    print(self.last_layer.layer_output)
                    raise IndexError("Index out of range")
                network_predictions.append(predicted_label)
    
                network_error = network_error + abs(predicted_label - train_outputs[sample_idx])

            self.update_weights(network_error)

    def feed_sample(self, sample):
        last_layer_outputs = sample
        for layer in self.network_layers:
            if type(layer) is Conv2D:
                layer.conv(input2D=last_layer_outputs)
            elif type(layer) is Dense:
                layer.dense_layer(layer_input=last_layer_outputs)
            elif type(layer) is MaxPooling2D:
                layer.max_pooling(input2D=last_layer_outputs)
            elif type(layer) is AveragePooling2D:
                layer.average_pooling(input2D=last_layer_outputs)
            elif type(layer) is ReLU:
                layer.relu_layer(layer_input=last_layer_outputs)
            elif type(layer) is Flatten:
                layer.flatten(input2D=last_layer_outputs)
            elif type(layer) is Input2D:
                pass
            else:
                print("Other")
                raise TypeError("The layer of type {layer_type} is not supported yet.".format(layer_type=type(layer)))

            last_layer_outputs = layer.layer_output
        return self.network_layers[-1].layer_output

    def update_weights(self, network_error):
        for layer in self.network_layers:
            if "trained_weights" in vars(layer).keys():
                layer.trained_weights = layer.trained_weights - network_error * self.learning_rate * layer.trained_weights

    def predict(self, data_inputs):
        if (data_inputs.ndim != 4):
            raise ValueError("The data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(num_dims=data_inputs.ndim))

        predictions = []
        for sample in data_inputs:
            probs = self.feed_sample(sample=sample)
            predicted_label = numpy.where(numpy.max(probs) == probs)[0][0]
            predictions.append(predicted_label)
        return predictions

    def summary(self):
        print("\n----------Network Architecture----------")
        for layer in self.network_layers:
            print(type(layer))
        print("----------------------------------------\n")
