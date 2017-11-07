using System.Threading.Tasks;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using SharedBenchmark;

namespace DigitsTest
{
    class Program
    {
        static async Task Main()
        {
            await MnistTester.PerformBenchmarkAsync(LearningAlgorithmType.BoundedBFGSWithGradientDescentOnFirstConvergence, 1000, false, 2000, false,
                NetworkLayer.Inputs(784),
                NetworkLayer.FullyConnected(60, ActivationFunctionType.Sigmoid),
                NetworkLayer.FullyConnected(16, ActivationFunctionType.Sigmoid),
                NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid));
        }
    }
}
