# MNIST Decision Tree

A decision tree of user specified depth to decide whether an image from a subset of the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a 3 or an 8.

To use, see `run.py`

Python (3.7) Modules required:
- numpy
- scipy
<!-- - matplotlib -->

If we know the value of a single pixel, how accurately can we classify the data? This decision tree determines the best points to split on at each depth, and uses the binary value of the pixel to split at each node.