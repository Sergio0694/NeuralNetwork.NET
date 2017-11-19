![](http://i.pi.gy/8ZDDE.png)

## What is it?
`NeuralNetwork.NET` is a .NET Standard 2.0 library that implements a Convolutional Neural Network with customizable layers, built from scratch with C#.
It provides simple APIs to define a CNN structure and to train the network using Stochastic Gradient Descent, as well as methods to save/load a network in JSON format and more.

There's also a secondary .NET Framework 4.7 library, `NeuralNetwork.NET.Cuda` that leverages the GPU to greatly increase the performances when training or using a neural network.

# Table of Contents

- [Installing from NuGet](#installing-from-nuget)
- [Quick start](#quick-start)
  - [Supervised learning](#supervised-learning) 
  - [Serialization and deserialization](#serialization-and-deserialization)
- [GPU acceleration](#gpu-acceleration)
- [Requirements](#requirements)

# Installing from NuGet

To install `NeuralNetwork.NET`, run the following command in the **Package Manager Console**

```
Install-Package NeuralNetwork.NET
```

More details available [here](https://www.nuget.org/packages/NeuralNetwork.NET/). (link not working yet)

# Quick start

### Supervised learning

Training a neural network is pretty straightforward - just use the methods in the `BackpropagationNetworkTrainer` class. For example, here's how to create and train a new neural network from scratch:

```C#
(INeuralNetwork network, _) = await NetworkTrainer.ComputeTrainedNetworkAsync(
  dataset, // A (float[,], float[,]) tuple with the training samples and labels
  10, // The expected number of training epochs to run
  100, // The size of each training mini-batch
  null, // An optional validation dataset for early-stopping
  null, // An optional test dataset to monitor the training progress
  CancellationToken.None, // An optional cancellation token for the training session
  NetworkLayers.FullyConnected(784, 100, ActivationFunctionType.Sigmoid),
  NetworkLayers.FullyConnected(100, 10, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
```

### Serialization and deserialization

The `INeuralNetwork` interface exposes a `SerializeAsJSON` method that can be used to serialize any network at any given time.
In order to get a new network instance from a serialized JSON string, just use the `NeuralNetworkDeserializer.TryDeserialize` method: it will parse the input text and automatically return a neural network with the original parameters.

# GPU acceleration

When using the `NeuralNetwork.NET.Cuda` additional library, it is possible to enable and disable the GPU acceleration at any time. To turn it on, just set the static property in the dedicated settings class:

```C#
NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
```

# Requirements

The `NeuralNetwork.NET` library requires .NET Standard 2.0 support, so it is available for applications targeting:
- .NET Framework >= 4.7
- .NET Core >= 2.0
- New versions of Mono and Xamarin.

The additional `NeuralNetwork.NET.Cuda` library requires .NET Framework >= 4.7 and a CUDA enabled GPU.
