using System;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda
{
    /// <summary>
    /// A static class for some additional settings for the <see cref="SupervisedLearning.BackpropagationNetworkTrainer"/> class
    /// </summary>
    public static class NetworkTrainerGpuPreferences
    {
        private static ProcessingMode _ProcessingMode = ProcessingMode.Cpu;

        /// <summary>
        /// Gets or sets the desired processing mode to perform the network training
        /// </summary>
        public static ProcessingMode ProcessingMode
        {
            get => _ProcessingMode;
            set
            {
                if (_ProcessingMode != value)
                {
                    switch (value)
                    {
                        case ProcessingMode.Cpu:
                            MatrixServiceProvider.ResetInjections();
                            break;
                        case ProcessingMode.Gpu:
                            MatrixServiceProvider.SetupInjections(
                                MatrixGpuExtensions.Multiply,
                                MatrixGpuExtensions.TransposeAndMultiply,
                                MatrixGpuExtensions.MultiplyAndSigmoid,
                                MatrixGpuExtensions.Sigmoid,
                                MatrixGpuExtensions.HalfSquaredDifference,
                                MatrixGpuExtensions.InPlaceSubtractAndHadamardProductWithSigmoidPrime,
                                MatrixGpuExtensions.InPlaceSigmoidPrimeAndHadamardProduct);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(value), value, null);
                    }
                    _ProcessingMode = value;
                }
            }
        }
    }
}
