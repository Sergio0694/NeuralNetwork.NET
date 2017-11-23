using System;
using NeuralNetworkNET.Cuda.Helpers;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda.APIs
{
    /// <summary>
    /// A static class for some additional settings for the neural networks produced by the library
    /// </summary>
    public static class NeuralNetworkGpuPreferences
    {
        private static ProcessingMode _ProcessingMode = ProcessingMode.Cpu;

        /// <summary>
        /// Gets or sets the desired processing mode to perform the neural network operations
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
                                MatrixGpuExtensions.MultiplyWithSum,
                                MatrixGpuExtensions.TransposeAndMultiply,
                                MatrixGpuExtensions.Activation,
                                MatrixGpuExtensions.MultiplyAndInPlaceActivationPrimeAndHadamardProduct,
                                ConvolutionGpuExtensions.ConvoluteForward,
                                ConvolutionGpuExtensions.ConvoluteBackwards,
                                ConvolutionGpuExtensions.ConvoluteGradient);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(value), value, null);
                    }
                    _ProcessingMode = value;
                }
            }
        }

        private static ulong _GPUMemoryAllocationLimit = UInt64.MaxValue;

        /// <summary>
        /// Gets or sets an additional limit on the amount of memory allocation to perform on the GPU memory
        /// </summary>
        public static ulong GPUMemoryAllocationLimit
        {
            get => _GPUMemoryAllocationLimit;
            set => _GPUMemoryAllocationLimit = value >= 1024 ? value : throw new ArgumentOutOfRangeException("Can't specify a limit less than 1KB");
        }
    }
}
