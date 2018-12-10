### LeNet Example

_Thanks to @Russell91 for this example_

This example showns you how to finetune code from the [Caffe MNIST tutorial](http://caffe.berkeleyvision.org/gathered/examples/mnist.html) using Tensorflow.
First, you can convert a prototxt model to tensorflow code:

    $ ./convert.py examples/mnist/lenet.prototxt --code-output-path=mynet.py

This produces tensorflow code for the LeNet network in `mynet.py`. The code can be imported as described below in the Inference section. Caffe-tensorflow also lets you convert `.caffemodel` weight files to `.npy` files that can be directly loaded from tensorflow:

    $ ./convert.py examples/mnist/lenet.prototxt --caffemodel examples/mnist/lenet_iter_10000.caffemodel --data-output-path=mynet.npy

The above command will generate a weight file named `mynet.npy`.

#### Inference:

Once you have generated both the code weight files for LeNet, you can finetune LeNet using tensorflow with

    $ ./examples/mnist/finetune_mnist.py

At a high level, `finetune_mnist.py` works as follows:

```python
# Import the converted model's class
from mynet import MyNet

# Create an instance, passing in the input data
net = MyNet({'data':my_input_data})

with tf.Session() as sesh:
    # Load the data
    net.load('mynet.npy', sesh)
    # Forward pass
    output = sesh.run(net.get_output(), ...)
```

#### Standalone model file:

You can save a standalone GraphDef model file as follows:

    $ ./convert.py examples/mnist/lenet.prototxt --caffemodel examples/mnist/lenet_iter_10000.caffemodel --standalone-output-path=mynet.pb

This generates a protobuf file named `mynet.pb` containing the model's graph and parameters. The [TensorFlow Image Recognition tutorial](https://www.tensorflow.org/versions/r0.11/tutorials/image_recognition/index.html) shows how to use models constructed in this way in [Python](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/image/imagenet) or [C++](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image).