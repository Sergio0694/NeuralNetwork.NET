using System.Runtime.CompilerServices;

// Used to inject the AleaGPU usage from the .NET Framework library
[assembly: InternalsVisibleTo("NeuralNetwork.NET.Cuda")]

// Allow the Unit tests to access internal methods
[assembly: InternalsVisibleTo("NeuralNetwork.NET.Unit")]
[assembly: InternalsVisibleTo("NeuralNetwork.NET.Cuda.Unit")]
