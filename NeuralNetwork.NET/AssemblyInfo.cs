using System.Runtime.CompilerServices;

// Allow the Unit tests to access internal methods
[assembly: InternalsVisibleTo("NeuralNetwork.NET.Unit")]

// Used to inject the AleaGPU usage from the .NET Framework library
[assembly:InternalsVisibleTo("NeuralNetwork.NET.Cuda")]
