![](http://i.pi.gy/8ZDDE.png)
[![NuGet](https://img.shields.io/nuget/v/NeuralNetwork.NET.svg)](https://www.nuget.org/packages/NeuralNetwork.NET/) [![NuGet](https://img.shields.io/nuget/dt/NeuralNetwork.NET.svg)](https://www.nuget.org/stats/packages/NeuralNetwork.NET?groupby=Version) [![Twitter Follow](https://img.shields.io/twitter/follow/Sergio0694.svg?style=social&label=Follow)](https://twitter.com/SergioPedri)

# What is it?

**NeuralNetwork.NET** is a .NET Standard 2.0 library that implements a Convolutional Neural Network with customizable layers, built from scratch with C#.
It provides simple APIs to define a CNN structure and to train the network using Stochastic Gradient Descent, as well as methods to save/load a network and its metadata and more.

The library also exposes CUDA-accelerated layers with more advanced features that leverage the GPU and the cuDNN toolkit to greatly increase the performances when training or using a neural network.

# Table of Contents

- [Installing from NuGet](#installing-from-nuget)
- [Quick start](#quick-start)
  - [Supervised learning](#supervised-learning) 
  - [GPU acceleration](#gpu-acceleration)
  - [Library settings](#library-settings)
  - [Serialization and deserialization](#serialization-and-deserialization)
  - [Built-in datasets](#built\-in-datasets)
- [Requirements](#requirements)

# Installing from NuGet

To install **NeuralNetwork.NET**, run the following command in the **Package Manager Console**

```
Install-Package NeuralNetwork.NET
```

More details available [here](https://www.nuget.org/packages/NeuralNetwork.NET/).

# Quick start

The **NeuralNetwork.NET** library exposes easy to use classes and methods to create a new neural network, prepare the datasets to use and train the network. These APIs are designed for rapid prototyping, and this section provides an overview of the required steps to get started.

## Supervised learning

The first step is to create a custom network structure. Here is an example with a sequential network (a stack of layers):

```C#
INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(28, 28),
    NetworkLayers.Convolutional((5, 5), 20, ActivationFunctionType.Identity),
    NetworkLayers.Pooling(ActivationFunctionType.LeakyReLU),
    NetworkLayers.Convolutional((3, 3), 40, ActivationFunctionType.Identity),
    NetworkLayers.Pooling(ActivationFunctionType.LeakyReLU),
    NetworkLayers.FullyConnected(125, ActivationFunctionType.LeakyReLU),
    NetworkLayers.FullyConnected(64, ActivationFunctionType.LeakyReLU),
    NetworkLayers.Softmax(10));
```

The next step is to prepare the datasets to use, through the APIs in the `DatasetLoader` class:

```C#
// A training dataset with a batch size of 100
IEnumerable<(float[] x, float[] u)> data = ... // Your own dataset parsing routine
ITrainingDataset dataset = DatasetLoader.Training(data, 100);

// An optional test dataset with a callback to monitor the progress
ITestDataset test = DatasetLoader.Test(..., new Progress<TrainingProgressEventArgs>(p =>
{
    Console.WriteLine($"Epoch {p.Iteration}, cost: {p.Cost}, accuracy: {p.Accuracy}");
});
```

Training a neural network is pretty straightforward - just use the methods in the `NetworkManager` class:

```C#    
// Train the network using Adadelta and 0.5 dropout probability
TrainingSessionResult result = NetworkManager.TrainNetwork(network, 
    dataset,                                // The ITrainingDataset instance   
    TrainingAlgorithms.AdaDelta(),          // The training algorithm to use
    60,                                     // The expected number of training epochs to run
    0.5f,                                   // Dropout probability
    new Progress<BatchProgress>(p => ...),  // Optional training epoch progress callback
    null,                                   // Optional callback to monitor the accuracy on the training dataset
    null,                                   // Optional validation dataset
    test,                                   // Test dataset
    token);                                 // Cancellation token for the training
```

**Note:** the `NetworkManager` methods are also available as asynchronous APIs.

## GPU acceleration

When running on a supported framework (.NET Framework, Xamarin or Mono), it is possible to use a different implementation of the available layers that leverages the cuDNN toolkit and parallelizes most of the work on the available CUDA-enabled GPU. To do that, just use the layers from the `CuDnnNetworkLayers` class when creating a network.

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

**NOTE:** in order to use this feature, the CUDA and cuDNN toolkits must be installed on the current system, a CUDA-enabled nVidia GeForce/Quadro GPU must be available and the **Alea** NuGet package must be installed in the application using the **NeuralNetwork.NET** library as well. Additional info are available [here](http://www.aleagpu.com/release/3_0_4/doc/installation.html#deployment_considerations).

## Library settings

**NeuralNetwork.NET** provides various shared settings that are available through the `NetworkSettings` class.
This class acts as a container to quickly check and modify any setting at any time, and these settings will influence the behavior of any existing `INeuralNetwork` instance and the library in general.

For example, it is possible to customize the criteria used by the networks to check their performance during training

```C#
NetworkSettings.AccuracyTester = AccuracyTesters.Argmax();      // The default mode (mutually-exclusive classes)

// Other testers are available too
NetworkSettings.AccuracyTester = AccuracyTesters.Threshold();   // Useful for overlapping classes
NetworkSettings.AccuracyTester = AccuracyTesters.Distance(0.2f); // Distance between results and expected outputs
```

When using CUDA-powered networks, sometimes the GPU in use might not be able to process the whole test or validation datasets in a single pass, which is the default behavior (these datasets are not divided into batches).
To avoid memory issues, it is possible to modify this behavior:

```C#
NetworkSettings.MaximumBatchSize = 400;   // This will apply to any test or validation dataset
```

## Serialization and deserialization

The `INeuralNetwork` interface exposes a `Save` method that can be used to serialize any network at any given time.
In order to get a new network instance from a saved file or stream, just use the `NetworkLoader.TryLoad` method.

As multiple layer types have different implementations across the available libraries, you can specify the layer providers to use when loading a saved network. For example, here's how to load a network using the cuDNN layers, when possible:

```C#
FileInfo file = new FileInfo(@"C:\...\MySavedNetwork.nnet");
INeuralNetwork network = NetworkLoader.TryLoad(file, LayersLoadingPreference.Cuda);
```

**Note:** the `LayersLoadingPreference` option indicates the desired type of layers to deserialize whenever possible. For example, using `LayersLoadingPreference.Cpu`, the loaded network will only have CPU-powered layers, if supported.

There's also an additional `SaveMetadataAsJson` method to export the metadata of an `INeuralNetwork` instance.

## Built-in datasets

The `NeuralNetworkNET.Datasets` namespace includes static classes to quickly load a popular dataset and get an `IDataset` instance ready to use with a new neural network. As an example, here's how to get the MNIST dataset:

```C#
ITrainingDataset trainingData = await Mnist.GetTrainingDatasetAsync(400); // Batches of 400 samples
ITestDataset testData = await Mnist.GetTestDatasetAsync(new Progress<TrainingProgressEventArgs>(p =>
{
    // Progress callback here
}));
```

Each API in this namespace also supports an optional `CancellationToken` to stop the dataset loading, as the source data is downloaded from the internet and can take some time to be available, depending on the dataset being used.

# Requirements

The **NeuralNetwork.NET** library requires .NET Standard 2.0 support, so it is available for applications targeting:
- .NET Framework >= 4.6.1
- .NET Core >= 2.0
- UWP (from SDK 10.0.16299)
- Mono 5.4
- Xamarin.iOS 10.14, Xamarin.Mac 3.8, Xamarin.Android 8.0

In addition to the frameworks above, you need an IDE with C# 7.2 support to compile the library on your PC.
