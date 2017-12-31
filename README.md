![](http://i.pi.gy/8ZDDE.png)

## What is it?
`NeuralNetwork.NET` is a .NET Standard 2.0 library that implements a Convolutional Neural Network with customizable layers, built from scratch with C#.
It provides simple APIs to define a CNN structure and to train the network using Stochastic Gradient Descent, as well as methods to save/load a network and its metadata and more.

The library also exposes Cuda-accelerated layers with more advanced features that leverage the GPU and the cuDNN toolkit to greatly increase the performances when training or using a neural network.

# Table of Contents

- [Quick start](#quick-start)
  - [Supervised learning](#supervised-learning) 
  - [GPU acceleration](#gpu-acceleration)
  - [Serialization and deserialization](#serialization-and-deserialization)
- [Requirements](#requirements)

# Quick start

### Supervised learning

Training a neural network is pretty straightforward - just use the methods in the `NetworkManager` class. For example, here's how to create and train a new neural network from scratch:

```C#
// A convolutional neural network to use with the MNIST dataset
INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.CreateForGrayscaleImage(28, 28),
    NetworkLayers.Convolutional((5, 5), 20, ActivationFunctionType.Identity),
    NetworkLayers.Pooling(ActivationFunctionType.LeakyReLU),
    NetworkLayers.Convolutional((3, 3), 40, ActivationFunctionType.Identity),
    NetworkLayers.Pooling(ActivationFunctionType.LeakyReLU),
    NetworkLayers.FullyConnected(125, ActivationFunctionType.LeakyReLU),
    NetworkLayers.FullyConnected(64, ActivationFunctionType.LeakyReLU),
    NetworkLayers.Softmax(10));
    
// Train the network using Adadelta and 0.5 dropout probability
TrainingSessionResult result = NetworkManager.TrainNetwork(network, 
    dataset, // A (float[,], float[,]) tuple with the training samples and labels
    60, // The expected number of training epochs to run
    100, // The size of each training mini-batch
    TrainingAlgorithmsInfo.Adadelta(), // The training algorithm to use
    0.5f, // Dropout probability
    new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
    {
        Printf($"Epoch {p.Iteration}, cost: {p.Cost}, accuracy: {p.Accuracy}");
    }))); // The test dataset to monitor the progress
```

**Note:** the `NetworkManager` methods are also available as asynchronous APIs.

### GPU acceleration

When running on a supported framework (.NET Framework, Xamarin or Mono), it is possible to use a different implementation of the available layers that leverages the cuDNN toolkit and parallelizes most of the work on the available CUDA-enabled GPU. To do that, just create a network using the layers from the `CuDnnNetworkLayers` class to enable the GPU processing mode.

Some of the cuDNN-powered layers support additional options than the default layers. Here's an example:

```C#
// A cuDNN convolutional layer, with custom mode, padding and stride
LayerFactory convolutional = CuDnnNetworkLayers.Convolutional(
    ConvolutionInfo.New(ConvolutionMode.CrossCorrelation, 2, 2),
    (5, 5), 20, ActivationFunctionType.ReLU);
    
// An inception module, from the design of the GoogLeNet network
LayerFactory inception = CuDnnNetworkLayers.Inception(InceptionInfo.New(
    10,     // 1x1 convolution kernels
    20, 10, // 1x1 + 3x3 convolution pipeline kernels
    20, 10, // 1x1 + 5x5 convolution pipeline kernels
    PoolingMode.AverageExcludingPadding, 10)); // Pooling mode and 1x1 convolution kernels
```

These `LayerFactory` instances can be used to create a new network just like in the CPU example.

### Serialization and deserialization

The `INeuralNetwork` interface exposes a `Save` method that can be used to serialize any network at any given time.
In order to get a new network instance from a saved file or stream, just use the `NeuralNetworkLoader.TryLoad` method.

As multiple layer types have different implementations across the available libraries, you can specify the layer providers to use when loading a saved network. For example, here's how to load a network using the cuDNN layers, when possible:

```C#
FileInfo file = new FileInfo(@"C:\...\MySavedNetwork.nnet");
INeuralNetwork network = NeuralNetworkLoader.TryLoad(file, CuDnnNetworkLayersDeserializer.Deserializer);
```

There's also an additional `SaveMetadataAsJson` method to export the metadata of an `INeuralNetwork` instance.

# Requirements

The `NeuralNetwork.NET` library requires .NET Standard 2.0 support, so it is available for applications targeting:
- .NET Framework >= 4.7.1
- .NET Core >= 2.0
- New versions of Mono and Xamarin.

In addition to the frameworks above, you need an IDE with C# 7.2 support to compile the library on your PC.

The cuDNN layers require .NET Framework, Xamarin or Mono and a CUDA enabled GPU.
