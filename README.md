![](http://i.pi.gy/8ZDDE.png)

## What is it?
`NeuralNetwork.NET` is a .NET Standard 2.0 library that implements various types of neural networks (multilayered, with and without bias and with different activation types) as well as multiple training methods, both for supervised and unsupervised learning.
It provides simple APIs to create and train neural networks given a cost and gradient function (supervised learning through backpropagation) or a user defined fitness function (unsupervised learning).

There's also a secondary .NET Framework 4.7 library, `NeuralNetwork.NET.Cuda` that leverages the GPU to greatly increase the performances when training or using a neural network.

# Table of Contents

- [Installing from NuGet](#installing-from-nuget)
- [Quick start](#quick-start)
  - [Supervised learning](#supervised-learning) 
  - [Activation functions](#activation-functions)
  - [Kernel convolutions](#kernel-convolutions)
  - [Unsupervised learning](#unsupervised-learning)
  - [Serialization and deserialization](#serialization-and-deserialization)
- [GPU acceleration](#gpu-acceleration)
- [Requirements](#requirements)

# Installing from NuGet

To install `NeuralNetwork.NET`, run the following command in the **Package Manager Console**

```
Install-Package NeuralNetwork.NET
```

More details available [here](https://www.nuget.org/packages/NeuralNetwork.NET/).

# Quick start

### Supervised learning

Training a neural network is pretty straightforward - just use the methods in the `BackpropagationNetworkTrainer` class. For example, here's how to create and train a new neural network from scratch:

```C#
INeuralNetwork network = await BackpropagationNetworkTrainer.ComputeTrainedNetworkAsync(
  x, // The [nsamples*inputneurons] dataset
  y, // The expected results to calculate the cost, a [nsamples*outputsize] matrix
  1000, // The size of each training mini batch
  LearningAlgorithmType.GradientDescent, // The training algorithm to use
  NeuralNetworkType.Biased, // The type of network to create
  token, // A CancellationToken for the training session
  new Progress<BackpropagationProgressEventArgs>(p =>
  {
      // Some callback code here
  }),
  784, 16, 16, 10); // The neurons in each network layer (here there are two 16-neurons hidden layers)
```

### Activation functions

It is possible to choose the activation function to use when training a network from the list of available activation functions exposed by the `ActivationFunction` enum. For example, to use a Tanh activation:

```C#
NeuralNetworkSettings.ActivationFunctionType = ActivationFunction.Tanh;
```

### Kernel convolutions

This library includes a customizable kernel convolution pipeline to process raw data before using it to train a neural network. Different kernel convolutions are available, as well as normalization methods, ReLU methods and more.
In order to process some data, first setup a pipeline:

```C#
ConvolutionPipeline pipeline = new ConvolutionPipeline( // Let's assume the source data matrix is 28*28
    ConvolutionOperation.Convolution3x3( 
        KernelsCollection.TopSobel,
        KernelsCollection.Outline,
        KernelsCollection.Sharpen,
        KernelsCollection.BottomLeftEmboss), // 4 3*3 kernels: 28*28*1 pixels >> 26*26*4
    ConvolutionOperation.ReLU, // Set minimum threshold
    ConvolutionOperation.Pool2x2, // 26*26*4 >> 13*13*4
    ConvolutionOperation.Normalization, // Normalize all the values in the [0..1] range
    ConvolutionOperation.Convolution3x3( 
        KernelsCollection.TopSobel,
        KernelsCollection.Outline)); // And so on...
```

Then use the pipeline to process data and get a single data matrix with all the results:

```C#
double[,] convolutions = pipeline.Process(dataset);
```

This processed matrix can now be used as data to train a network using the backpropagation APIs.

### Unsupervised learning

The library provides a `NeuralNetworkGeneticAlgorithmProvider` class that implements a genetic algorithm. This class can be initialized using different parameters and will run the algorithm to create and train the neural networks.
First, declare a fitness function using the `FitnessDelegate` delegate.
This delegate takes as arguments an identifier for the current network and its forward function, and returns the fitness score for the tested species.
It also provides a list of the forward functions for the other species in the current generation: this can be used to test each network against all the other ones to get some sort of competition.
The list is created using the lazy evaluation of the LINQ library, so it doesn't use CPU time if it's not used in the body of the fitness function.

```C#
FitnessDelegate fitnessFunction = (uid, f, opponents) =>
{
  // The uid parameter is a unique uid for the current neural network calling the fitness function
  double[,] testData = PrepareTestData(); // Prepare your own data to feed the neural network
  double[,] result = f(testData);

  // Calculate the score using the result array and return the fitness value
  return CalculateFitness(result);
};
```

Then get a neural network provider using one of the available methods:

```C#
NeuralNetworkGeneticAlgorithmProvider provider = await NeuralNetworkGeneticAlgorithmProvider.NewSingleLayerPerceptronProviderAsync(
  fitnessFunction, // The fitness function to test the networks
  16, // Number of inputs
  4, // Number of outputs
  16, // Number of neurons in the hidden layer
  100, // Size of the population for the genetic algorithm
  5, // Percentage of random mutations for each weight in the networks
  10); // Number of best networks to carry over each generation
```

You can also use a callback action to monitor the provider feedback:

```C#
IProgress<GeneticAlgorithmProgress> callback = new Progress<GeneticAlgorithmProgress>(p =>
{
  // Display the progress
  Console.Writeline($"Generation {p.Generation}, best: {p.Best}, average: {p.Average}, all time best score: {p.AllTimeBest}");
});
provider.ProgressCallback = callback;
```
    
Then you can start and stop the provider when necessary:

```C#
// Returns true if the provider is started correctly, false if it was already running
provider.StartAsync();

// Wait until a good enough network has been trained
provider.StopAsync();
```
    
It is now possible to get the network and use it:

```C#
INeuralNetwork network = provider.BestNetwork;
```
    
The `INeuralNetwork` interface has a `Forward` method that can be used to process some input data with the network:

```C#
double[,] result = network.Forward(input);
```

### Serialization and deserialization

The `INeuralNetwork` interface exposes a `SerializeAsJSON` method that can be used to serialize any network at any given time.
In order to get a new network instance from a serialized JSON string, just use the `NeuralNetworkDeserializer.TryDeserialize` method: it will parse the input text and automatically return a neural network with the original parameters.

# GPU acceleration

When using the `NeuralNetwork.NET.Cuda` additional library, it is possible to enable and disable the GPU acceleration at any time. To turn it on, just set the static property in the dedicated settings class:

```C#
NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
```

This will enable the GPU acceleration both for the network training and the kernel convolution pipeline processing.

**Note**: to make sure not to exceed the available GPU memory, it is possible to set an explicit memory threshold:

```C#
NeuralNetworkGpuPreferences.GPUMemoryAllocationLimit = 800_000_000; // ~800MB
```

# Requirements

The `NeuralNetwork.NET` library requires .NET Standard 2.0 support, so it is available for applications targeting .NET Framework >= 4.7, .NET Core >= 2.0 and new versions of Mono and Xamarin.

The additional `NeuralNetwork.NET.Cuda` library requires .NET Framework >= 4.7 and a CUDA enabled GPU.



