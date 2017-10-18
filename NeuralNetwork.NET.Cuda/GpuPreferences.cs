using System;
using Alea;
using Alea.Parallel;
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
                            MatrixExtensions.MultiplyOverride = CudaMultiply;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(value), value, null);
                    }
                    _ProcessingMode = value;
                }
            }
        }

        private static double[,] CudaMultiply(double[,] m1, double[,] m2)
        {
            int m = m1.GetLength(0);
            int n = m2.GetLength(1);
            int p = m2.GetLength(1);
            double[,] result = new double[m, p];
            

            Gpu.Default.For(0, m * p, index =>
            {
                int
                    i = index / p,
                    j = index % p;

                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += m1[i, k] * m2[k, j];
                }
                result[i, j] = sum;
            });
            return result;
        }
    }
}
