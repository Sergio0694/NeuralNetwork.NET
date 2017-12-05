![](http://i.pi.gy/8ZDDE.png)

## What is it?
`NeuralNetwork.NET` is a .NET Standard 2.0 library that implements a Convolutional Neural Network with customizable layers, built from scratch with C#.
It provides simple APIs to define a CNN structure and to train the network using Stochastic Gradient Descent, as well as methods to save/load a network in JSON/binary format and more.

There's also a secondary .NET Framework 4.7 library, `NeuralNetwork.NET.Cuda` that leverages the GPU to greatly increase the performances when training or using a neural network.

# Table of Contents

- [Quick start](#quick-start)
  - [Supervised learning](#supervised-learning) 
  - [Serialization and deserialization](#serialization-and-deserialization)
- [GPU acceleration](#gpu-acceleration)
- [Requirements](#requirements)

# Quick start

### Supervised learning

Training a neural network is pretty straightforward - just use the methods in the `NetworkManager` class. For example, here's how to create and train a new neural network from scratch:

```C#
// A simple network to use with the MNIST dataset
INeuralNetwork network = NetworkManager.NewNetwork(
    NetworkLayers.Convolutional((28, 28, 1), (5, 5), 10, ActivationFunctionType.Identity),
    NetworkLayers.Pooling((24, 24, 10), ActivationFunctionType.Sigmoid),
    NetworkLayers.FullyConnected(12 * 12 * 10, 100, ActivationFunctionType.Sigmoid),
    NetworkLayers.Softmax(100, 10));
    
TrainingSessionResult result = NetworkManager.TrainNetwork(network, 
    (training.X, training.Y), // A (float[,], float[,]) tuple with the training samples and labels
    60, // The expected number of training epochs to run
    10, // The size of each training mini-batch
    null, // An optional validation dataset for early-stopping
    new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
    {
        Printf($"Epoch {p.Iteration}, cost: {p.Cost}, accuracy: {p.Accuracy}");
    })), // The test dataset to monitor the progress
    0.1f, // The learning rate
    0.5f, // The dropout probability for the fully connected layers
    0.025f); // The optional L2 regularization parameter
```

**Note:** the `NetworkManager` methods are also available as asynchronous APIs.

### Serialization and deserialization

The `INeuralNetwork` interface exposes a `SerializeAsJson` method that can be used to serialize any network at any given time.
In order to get a new network instance from a serialized JSON string, just use the `NeuralNetworkLoader.TryLoadJson` method: it will parse the input text and automatically return a neural network with the original parameters.

There's also an additional `Save` method to save a neural network to a binary file. This provides a small, easy to share file that contains all the info on the current network.

# GPU acceleration

When using the `NeuralNetwork.NET.Cuda` additional library, it is possible to enable and disable the GPU acceleration at any time. To turn it on, just set the static property in the dedicated settings class:

```C#
NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
```

# Requirements

The `NeuralNetwork.NET` library requires .NET Standard 2.0 support, so it is available for applications targeting:
- .NET Framework >= 4.7.1
- .NET Core >= 2.0
- New versions of Mono and Xamarin.

In addition to the frameworks above, you need an IDE with C# 7.2 support to compile the library on your PC.

The `NeuralNetwork.NET.Cuda` library requires .NET Framework >= 4.71 and a CUDA enabled GPU.
