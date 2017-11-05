![](http://i.pi.gy/8ZDDE.png)

## What is it?
`NeuralNetwork.NET` is a .NET Standard 2.0 library that implements various types of neural networks (multilayered, with and without bias and with different activation types) as well as multiple training methods, both for supervised and unsupervised learning.
It provides simple APIs to create and train neural networks given a cost and gradient function (supervised learning through backpropagation) or a user defined fitness function (unsupervised learning).

There's also a secondary .NET Framework 4.7 library, `NeuralNetwork.NET.Cuda` that leverages the GPU to greatly increase the performances when training or using a neural network.

## Usage

### Setup and compatibility

The library needs to be initialized with a wrapper for the `System.Threading.Tasks.Parallel.For` method, since it can't be referenced from a .NET Standard 1.4 project. To do so, just pass a `ParallelFor` delegate to the library, that will forward the call to the actual `Parallel.For` method:

```C#
ParallelCompatibilityWrapper.Initialize((start, end, body) => Parallel.For(start, end, body).IsCompleted);
```

Then, since the library can't reference the `Accord.Math` or the `portable.accord.math` NuGet packages on its own, a wrapper for the LBFGS method is also needed. You'll need to implement a class with the `IAccordNETGradientOptimizationMethod` interface (you'll find additional info in the file) and then call:

```C#
AccordNETGradientOptimizationMethodCompatibilityWrapper.Initialize(myLBFGSwrapperInstance);
```

### Supervised learning (backpropagation)

Training a neural network is pretty straightforward - just use the methods in the `GradientDescentNetworkTrainer` class. For example, here's how to create and train a `SingleLayerPerceptron` network instance:

```C#
INeuralNetwork network = await GradientDescentNetworkTrainer.ComputeTrainedNetworkAsync(
  inputs, // A [nsamples*hiddenlayersize] matrix
  y, // The expected results to calculate the cost, a [nsamples*outputsize] matrix
  90, // The number of neurons in the hidden layer (will be calculated automatically if null)
  token, // A cancellation token for the training session 
  null, // The optional starting solution for the training (the serialized weights from another network)
  new Progress<BackpropagationProgressEventArgs>(p =>
  {
      // Some callback code here
  }));
```

### Kernel convolutions

This library includes a customizable kernel convolution pipeline to process raw data before using it to train a neural network. Different kernel convolutions are available, as well as normalization methods, ReLU methods and more.
In order to process some data, first setup a pipeline:

```C#
ConvolutionPipeline pipeline = new ConvolutionPipeline( // Let's assume the source data matrix is 28*28
    v => v.Expand(ConvolutionExtensions.Convolute3x3, 
        KernelsCollection.TopSobel,
        KernelsCollection.Outline,
        KernelsCollection.Sharpen,
        KernelsCollection.BottomLeftEmboss), // 4 3*3 kernels: 28*28*1 pixels >> 26*26*4
    v => v.Process(ConvolutionExtensions.ReLU), // Set minimum threshold
    v => v.Process(ConvolutionExtensions.Pool2x2), // 26*26*4 >> 13*13*4
    v => v.Process(ConvolutionExtensions.Normalize), // Normalize all the values in the [0..1] range
    v => v.Expand(ConvolutionExtensions.Convolute3x3, 
        KernelsCollection.TopSobel,
        KernelsCollection.Outline)); // And so on...
```

Then use the pipeline to process data and get a single data matrix with all the results (each `ConvolutionsStack` entry in the processed list will be a single row in the final data matrix, and all of its values will be flattened in the matrix columns):

```C#
IReadOnlyList<ConvolutionsStack> convolutions = pipeline.Process(source);
double[,] inputs = ConvolutionPipeline.ConvertToMatrix(convolutions.ToArray());
```

This `double[,]` object can now be used as data to train a network using the backpropagation APIs.

### Unsupervised learning (genetic algorithm)

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
