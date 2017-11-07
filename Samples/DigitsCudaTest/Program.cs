using System.Threading.Tasks;
using NeuralNetworkNET.Cuda.APIs;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using SharedBenchmark;

namespace DigitsCudaTest
{
    class Program
    {
        static async Task Main()
        {
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
            await MnistTester.PerformBenchmarkAsync(LearningAlgorithmType.BoundedBFGSWithGradientDescentOnFirstConvergence, 1000, false, null, false, 
                NetworkLayer.Inputs(784), 
                NetworkLayer.FullyConnected(16, ActivationFunctionType.Sigmoid), 
                NetworkLayer.FullyConnected(16, ActivationFunctionType.Sigmoid), 
                NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid));
        }
    }
}
