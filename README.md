# NeuralNetworkLibrary

A simple library that implements a genetic algorithm to produce neural networks to perform various tasks.
This library provides simple APIs to create and train neural networks given a user defined fitness function.

## Usage

The library provides a `NeuralNetworkGeneticAlgorithmProvider` class that implements a genetic algorithm. This class can be initialized using different parameters and will run the algorithm to create and train the neural networks.
First, declare a fitness function using the `NeuralNetworkGeneticAlgorithmProvider.FitnessDelegate` delegate.
This delegate takes as arguments an identifier for the current network and its forward function, and returns the fitness score for the tested species.
It also provides a list of the forward functions for the other species in the current generation: this can be used to test each network against all the other ones to get some sort of competition.
The list is created using the lazy evaluation of the LINQ library, so it doesn't use CPU time if it's not used in the body of the fitness function.

```C#
NeuralNetworkGeneticAlgorithmProvider.FitnessDelegate fitnessFunction = (uid, f, opponents) =>
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
NeuralNetworkGeneticAlgorithmProvider provider = await NeuralNetworkGeneticAlgorithmProvider.NewSingleLayerAsync(
  fitnessFunction, // The fitness function to test the networks
  16, // Number of inputs
  4, // Number of outputs
  16, // Number of neurons in the hidden layer
  null, // Optional threshold for the activation function of the hidden layer neurons
  0.5, // Optiona threshold for the activation function of the output neurons
  100, // Size of the population for the genetic algorithm
  5, // Percentage of random mutations for each weight in the networks
  10); // Number of best networks to carry over each generation
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
    
It is also possible to serialize a network and restore it:

```C#
byte[] serialized = network.Serialize();

// The serialized data can be used to recreate the network or another provider
INeuralNetwork deserialized = NeuralNetworkDeserializer.TryGetInstance(serialized);
NeuralNetworkGeneticAlgorithmProvider provider = await NeuralNetworkGeneticAlgorithmProvider.FromSerializedNetworkAsync(
  fitnessFunction, // Use the same fitness function used to obtain the serialized network
  deserialized, // The serialized network (all the provider parameters will be extracted from the network info)
  100, // Population size
  5, // Random mutations probability
  10); // Elite samples mantained over each generation
```
