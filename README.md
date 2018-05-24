# NumPyCNN
Convolutional neural network implementation using NumPy. Just three layers are created which are convolution (conv for short), ReLU, and max pooling. The major steps involved are as follows:
1.	Reading the input image.
2.	Preparing filters.
3.	Conv layer: Convolving each filter with the input image.
4.	ReLU layer: Applying ReLU activation function on the feature maps (output of conv layer).
5.	Max Pooling layer: Applying the pooling operation on the output of ReLU layer.
6.	Stacking conv, ReLU, and max pooling layers

The file named **example.py** is an example of using the project.
The code starts by reading an input image. That image can be either single or multi-dimensional image.

```python
# Reading the image
#img = skimage.io.imread("test.jpg")
#img = skimage.data.checkerboard()
img = skimage.data.chelsea()
#img = skimage.data.camera()
```

In this examplel, an input gray is used and this is why it is required to ensure the image is already gray.
```python
# Converting the image into gray.
img = skimage.color.rgb2gray(img)
```

The filters of the first conv layer are prepared according to the input image dimensions. The filter is created by specifying the following:
1) Number of filters.
2) Size of first dimension.
3) Size of second dimension.
3) Size of third dimension and so on.

Because the previous image is just gray, then the filter will have just width and height and no depth. That is why it is created by specifying just three numbers (number of filters, width, and height). Here is an example of creating two 3x3 filters.
```python
# First conv layer
#l1_filter = numpy.random.rand(2,7,7)*20 # Preparing the filters randomly.
l1_filter = numpy.zeros((2,3,3))
l1_filter[0, :, :] = numpy.array([[[-1, 0, 1], 
                                   [-1, 0, 1], 
                                   [-1, 0, 1]]])
l1_filter[1, :, :] = numpy.array([[[1,   1,  1], 
                                   [0,   0,  0], 
                                   [-1, -1, -1]]])
```

The code can still work with RGb images. The only difference is using filters of similar shape to the image. If the image is RGB and not converted to gray, then the filter will be created by specifying 4 numbers (number of filters, width, height, and number of channels). Here is an example of creating two 7x7x3 filters.
```python
# First conv layer
l1_filter = numpy.random.rand(2, 7, 7, 3) # Preparing the filters randomly.
```

Next is to forward the filters to get applied on the image using the stack of layers used in the ConvNet.
```python
print("\n**Working with conv layer 1**")
l1_feature_map = NumPyCNN.conv(img, l1_filter)
print("\n**ReLU**")
l1_feature_map_relu = NumPyCNN.relu(l1_feature_map)
print("\n**Pooling**")
l1_feature_map_relu_pool = NumPyCNN.pooling(l1_feature_map_relu, 2, 2)
print("**End of conv layer 1**\n")
```

Here is the outputs of such conv-relu-pool layers.
![l1](https://user-images.githubusercontent.com/16560492/39051349-ac56ac56-44a8-11e8-8695-29901dd3a811.png)

```python
# Second conv layer
l2_filter = numpy.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 2**")
l2_feature_map = NumPyCNN.conv(l1_feature_map_relu_pool, l2_filter)
print("\n**ReLU**")
l2_feature_map_relu = NumPyCNN.relu(l2_feature_map)
print("\n**Pooling**")
l2_feature_map_relu_pool = NumPyCNN.pooling(l2_feature_map_relu, 2, 2)
print("**End of conv layer 2**\n")
```
The outputs of such conv-relu-pool layers are shown below.
![l2](https://user-images.githubusercontent.com/16560492/39051582-6abe0996-44a9-11e8-88e1-589a673a8b11.png)

```python
# Third conv layer
l3_filter = numpy.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 3**")
l3_feature_map = NumPyCNN.conv(l2_feature_map_relu_pool, l3_filter)
print("\n**ReLU**")
l3_feature_map_relu = NumPyCNN.relu(l3_feature_map)
print("\n**Pooling**")
l3_feature_map_relu_pool = NumPyCNN.pooling(l3_feature_map_relu, 2, 2)
print("**End of conv layer 3**\n")
```
The following graph shows the outputs of the above conv-relu-pool layers.
![l3](https://user-images.githubusercontent.com/16560492/39051603-76339f3e-44a9-11e8-8e4e-9303a51aaa79.png)

For more info.:
KDnuggets: https://www.kdnuggets.com/author/ahmed-gad
LinkedIn: https://www.linkedin.com/in/ahmedfgad
Facebook: https://www.facebook.com/ahmed.f.gadd
ahmed.f.gad@gmail.com
ahmed.fawzy@ci.menofia.edu.eg
