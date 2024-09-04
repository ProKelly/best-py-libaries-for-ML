# Best Python Libraries for Machine Learning

**Last Updated:** 08 Aug, 2024

Machine Learning, as the name suggests, is the science of programming a computer to learn from various types of data. Arthur Samuel defines it as:

> “Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed.”

In the early days, people manually coded all the algorithms, making the process time-consuming and inefficient. Today, Python has become one of the most popular programming languages for Machine Learning due to its extensive collection of libraries. Below are some of the key Python libraries used in Machine Learning:

- **Numpy**
- **SciPy**
- **Scikit-learn**
- **Theano**
- **TensorFlow**
- **Keras**
- **PyTorch**
- **Pandas**
- **Matplotlib**

## Numpy

**NumPy** is a popular Python library for large multi-dimensional array and matrix processing, aided by a collection of high-level mathematical functions. It is particularly useful for linear algebra, Fourier transform, and random number capabilities. Libraries like TensorFlow use NumPy internally for tensor manipulation.

### Example Code
```python
import numpy as np

# Creating two arrays of rank 2
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

# Creating two arrays of rank 1
v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors
print(np.dot(v, w), "\n")

# Matrix and Vector product
print(np.dot(x, v), "\n")

# Matrix and matrix product
print(np.dot(x, y))

```

## SciPy

SciPy is widely used in Machine Learning for tasks involving optimization, linear algebra, integration, and statistics. It is also useful for image manipulation.
### Exmaple of code 
``` python
from scipy.misc import imread, imsave, imresize

# Read a JPEG image into a numpy array
img = imread('D:/Programs/cat.jpg')
print(img.dtype, img.shape)

# Tinting the image
img_tint = img * [1, 0.45, 0.3]

# Saving the tinted image
imsave('D:/Programs/cat_tinted.jpg', img_tint)

# Resizing the tinted image to be 300 x 300 pixels
img_tint_resize = imresize(img_tint, (300, 300))

# Saving the resized tinted image
imsave('D:/Programs/cat_tinted_resized.jpg', img_tint_resize)
```

## Scikit-learn

Scikit-learn is one of the most popular libraries for classical Machine Learning algorithms. It is built on top of NumPy and SciPy, supporting supervised and unsupervised learning algorithms, data-mining, and data-analysis.

### Example of code 
``` python 
from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
dataset = datasets.load_iris()

# Fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# Make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# Summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```
![Screenshot from 2024-09-04 08-46-37](https://github.com/user-attachments/assets/445cb058-7ff7-45e8-a0e5-f44a21872cdc)

## Theano

Theano is used to define, evaluate, and optimize mathematical expressions involving multi-dimensional arrays. It optimizes the utilization of CPU and GPU and is widely used for unit-testing and error detection.

### Example of code 
``` python
import theano
import theano.tensor as T

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)

logistic([[0, 1], [-1, -2]])

```
#### output 
```
array([[0.5, 0.73105858], [0.26894142, 0.11920292]])
```

## TensorFlow

TensorFlow is an open-source library for high-performance numerical computation developed by Google Brain. It is widely used in deep learning research and applications.

### Example of code 
``` python
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])

# Multiply
result = tf.multiply(x1, x2)

# Initialize the Session
sess = tf.Session()

# Print the result
print(sess.run(result))

# Close the session
sess.close()
```

#### Output 
```
[ 5 12 21 32]
```

## Keras

Keras is a high-level neural networks API capable of running on TensorFlow, CNTK, or Theano. It is designed to enable fast experimentation with deep neural networks.

## PyTorch

PyTorch is a popular open-source Machine Learning library for Python based on Torch. It supports computations on Tensors with GPU acceleration and helps in creating computational graphs.

### Example code 
``` python
import torch

dtype = torch.float
device = torch.device("cpu")

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```
#### Output 
```
0 47168344.0
1 46385584.0
...
499 3.897604619851336e-05
```

## Pandas

Pandas is a popular Python library for data analysis, providing high-level data structures and tools for data manipulation. It is particularly useful for data extraction and preparation.

### Example code 
``` python
import pandas as pd

data = {
    "country": ["Brazil", "Russia", "India", "China", "South Africa"],
    "capital": ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
    "area": [8.516, 17.10, 3.286, 9.597, 1.221],
    "population": [200.4, 143.5, 1252, 1357, 52.98]
}

data_pandas = pd.DataFrame(data)
print(data_pandas)
```

#### Output 
```
        country    capital    area  population
0       Brazil   Brasilia   8.516      200.40
1       Russia    Moscow   17.10      143.50
2        India New Delhi   3.286     1252.00
3        China   Beijing   9.597     1357.00
4 South Africa  Pretoria   1.221       52.98
```

## Matplotlib

Matplotlib is a 2D plotting library in Python used for creating graphs and figures. It provides an object-oriented API to embed plots in applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK.

### Example code 
``` python
import matplotlib.pyplot as plt
import numpy as np

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()
```

## Conclusion

Python’s extensive range of libraries makes it a powerful tool for Machine Learning. The libraries discussed above are just a few of the many available. Choosing the right libraries can significantly impact the development and performance of Machine Learning models.

