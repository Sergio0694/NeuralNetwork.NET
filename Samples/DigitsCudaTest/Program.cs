using System.Threading.Tasks;
using NeuralNetworkNET.Cuda.APIs;
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
            await MnistTester.PerformBenchmarkAsync(LearningAlgorithmType.BoundedBFGSWithGradientDescentOnFirstConvergence, NeuralNetworkType.Biased, 1000, false, null, false, 784, 16, 16, 10);
        }
    }
}
