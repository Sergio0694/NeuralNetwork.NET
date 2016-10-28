# NeuralNetworkLibrary

A simple library that implements a genetic algorithm to produce neural networks to perform various tasks.
This library provides simple APIs to create and train neural networks given a user defined fitness function.

## Usage

The library provides a `NeuralNetworkGeneticAlgorithmProvider` class that implements a genetic algorithm. This class can be initialized using different parameters and will run the algorithm to create and train the neural networks.
First, declare a fitness function:

```C#
Func<int, Func<double[,], double[,]>, double> fitnessFunction = (uid, f) =>
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
NeuralNetworkGeneticAlgorithmProvider provider = await NeuralNetworkGeneticAlgorithmProvider.NewSingleLayerAsync(fitnessFunction, 16, 4, 16, null, 0.5, 100, 5, 10);
```

You can also use a callback action to monitor the provider feedback:

```C#
Action<GeneticAlgorithmProgress> callback = p =>
{
  // Display the progress
  Console.Writeline($"Generation {p.Generation}, best: {p.Best}, average: {p.Average}, all time best score: {p.AllTimeBest}");
};
provider.ProgressCallback = callback;
```
    
Then you can start and stop the provider when necessary:

```C#
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
    
It is also possible to serialize a network and restore it:

```C#
byte[] serialized = network.Serialize();

// The serialized data can be used to recreate the network or another provider
INeuralNetwork deserialized = NeuralNetworkDeserializer.TryGetInstance(serialized);
NeuralNetworkGeneticAlgorithmProvider provider = await NeuralNetworkGeneticAlgorithmProvider.FromSerializedNetworkAsync(fitnessFunction, deserialized, 100, 5, 10);
```
