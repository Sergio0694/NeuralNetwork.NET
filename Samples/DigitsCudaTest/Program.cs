using System.Threading.Tasks;
using NeuralNetworkNET.Cuda;
using NeuralNetworkNET.SupervisedLearning.Misc;
using SharedBenchmark;

namespace DigitsCudaTest
{
    class Program
    {
        static async Task Main()
        {
            NetworkTrainerGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
            await MnistTester.PerformBenchmarkAsync(LearningAlgorithmType.BoundedBFGSWithGradientDescentOnFirstConvergence, false, null, true, 784, 16, 16, 10);
        }
    }
}
