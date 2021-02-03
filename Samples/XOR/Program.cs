using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using System;
using System.Linq;

namespace XOR
{
    class Program
    {
        // hot-encoded values for true and false
        static readonly float[] FALSE = new[] { 0f, 1f };
        static readonly float[] TRUE = new[] { 1f, 0f };

        static void Main(string[] args)
        {
            var network = NetworkManager.NewSequential(TensorInfo.Linear(2),
                NetworkLayers.FullyConnected(2, ActivationType.Sigmoid),
                NetworkLayers.Softmax(2));

            var input = new[] { new[]{ 0f, 0f }, new[] { 1f, 0f }, new[] { 0f, 1f }, new[] { 1f, 1f } };
            var output = new[] { FALSE, TRUE, TRUE, FALSE };

            var trainingData = Enumerable.Zip(input, output).ToArray();
            var dataset = DatasetLoader.Training(trainingData, 300);

            NetworkManager.TrainNetworkAsync(network, dataset,
                TrainingAlgorithms.AdaDelta(), 3000).Wait();

            Test(network, new[] { 0f, 0f });
            Test(network, new[] { 0f, 1f });
            Test(network, new[] { 1f, 0f });
            Test(network, new[] { 1f, 1f });
        }

        static void Test(INeuralNetwork network, float[] input) {
            Console.Write($"{input[0]} XOR {input[1]} = ");
            var res = network.Forward(input);
            Console.WriteLine(res[0] > res[1]);
        }
    }
}
