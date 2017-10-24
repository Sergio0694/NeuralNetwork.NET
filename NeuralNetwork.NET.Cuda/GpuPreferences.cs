using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda
{
    public static class GpuPreferences
    {
        private static ProcessingMode _ProcessingMode = ProcessingMode.Cpu;

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
                            MatrixExtensions.MultiplyOverride = null;
                            break;
                        case ProcessingMode.Gpu:
                         //   MatrixExtensions.MultiplyOverride = CudaMultiply;
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
