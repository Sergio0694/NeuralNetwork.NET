using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Convolution
{
    public static class ConvolutionExtensions
    {
        /// <summary>
        /// Returns the normalized matrix with a max value of 1
        /// </summary>
        /// <param name="m">The input matrix to normalize</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Normalize([NotNull] this double[,] m)
        {
            // Prepare the result matrix
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[h, w];

            // Pool the input matrix
            unsafe
            {
                fixed (double* p = m, r = result)
                {
                    // Get the max value
                    double max = 0;
                    for (int i = 0; i < m.Length; i++)
                        if (p[i] > max) max = p[i];

                    // Normalize the matrix content
                    for (int i = 0; i < m.Length; i++)
                        r[i] = p[i] / max;
                }
            }
            return result;
        }

        /// <summary>
        /// Pools the input matrix with a window of 2 and a stride of 2
        /// </summary>
        /// <param name="m">The input matrix to pool</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Pool2x2([NotNull] this double[,] m)
        {
            // Prepare the result matrix
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[h / 2, w / 2];

            // Pool the input matrix
            int x = 0;
            for (int i = 0; i < h - 1; i += 2)
            {
                int y = 0;
                for (int j = 0; j < w - 1; j += 2)
                {
                    double
                        maxUp = m[i, j] > m[i, j + 1] ? m[i, j] : m[i, j + 1],
                        maxDown = m[i + 1, j] > m[i + 1, j + 1] ? m[i + 1, j] : m[i + 1, j + 1],
                        max = maxUp > maxDown ? maxUp : maxDown;
                    result[x, y++] = max;
                }
                x++;
            }
            return result;
        }

        /// <summary>
        /// Performs the Rectified Linear Units operation on the input matrix (applies a minimum value of 0)
        /// </summary>
        /// <param name="m">The input matrix to edit</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] ReLU([NotNull] this double[,] m)
        {
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[h, w];
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    result[i, j] = m[i, j] >= 0 ? m[i, j] : 0;
            return result;
        }

        /// <summary>
        /// Convolutes the input matrix with the given 3x3 kernel
        /// </summary>
        /// <param name="m">The input matrix</param>
        /// <param name="kernel">The 3x3 convolution kernel to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Convolute3x3([NotNull] this double[,] m, [NotNull] double[,] kernel)
        {
            // Prepare the output matrix
            if (kernel.Length != 9) throw new ArgumentOutOfRangeException("The input kernel must be 3x3");
            int h = m.GetLength(0), w = m.GetLength(1);
            if (h < 3 || w < 3) throw new ArgumentOutOfRangeException("The input matrix must be at least 3x3");
            double[,] result = new double[h - 2, w - 2];

            // Calculate the normalization factor
            double Abs(double value) => value >= 0 ? value : -value;
            double factor = 0;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    factor += Abs(kernel[i, j]);

            // Process the convolution
            int x = 0;
            for (int i = 1; i < h - 1; i++)
            {
                int y = 0;
                for (int j = 1; j < w - 1; j++)
                {
                    double
                        partial =
                            m[i - 1, j - 1] * kernel[0, 0] +
                            m[i - 1, j] * kernel[0, 1] +
                            m[i - 1, j + 1] * kernel[0, 2] +
                            m[i, j - 1] * kernel[1, 0] +
                            m[i, j] * kernel[1, 1] +
                            m[i, j + 1] * kernel[1, 2] +
                            m[i + 1, j - 1] * kernel[2, 0] +
                            m[i + 1, j] * kernel[2, 1] +
                            m[i + 1, j + 1] * kernel[2, 2],
                        normalized = partial / factor;
                    result[x, y++] = normalized;
                }
                x++;
            }
            return result;
        }
    }
}
