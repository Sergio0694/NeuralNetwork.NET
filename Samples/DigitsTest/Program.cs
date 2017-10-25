using System.Threading.Tasks;
using NeuralNetworkNET.SupervisedLearning.Misc;
using SharedBenchmark;

namespace DigitsTest
{
    class Program
    {
        static async Task Main()
        {
            await Helper.PerformBenchmarkAsync(LearningAlgorithmType.BoundedBFGSWithGradientDescentOnFirstConvergence, true, 2000, true, 480, 60, 16, 10);
        }
    }
}
