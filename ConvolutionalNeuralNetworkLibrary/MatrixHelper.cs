using System;
using JetBrains.Annotations;

namespace ConvolutionalNeuralNetworkLibrary
{
    /// <summary>
    /// An helper class with methods to process fixed-size matrices
    /// </summary>
    internal static class MatrixHelper
    {
        #region CNN

        /// <summary>
        /// Pools the input matrix with a window of 2 and a stride of 2
        /// </summary>
        /// <param name="m">The input matrix to pool</param>
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Pool2x2([NotNull] this double[,] m)
        {
            // Prepare the result matrix
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[h / 2, w / 2];

            // Pool the input matrix
            int x = 0;
            for (int i = 0; i < h; i += 2)
            {
                int y = 0;
                for (int j = 0; j < w; j += 2)
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
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void ReLU([NotNull] this double[,] m)
        {
            int h = m.GetLength(0), w = m.GetLength(1);
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    if (m[i, j] < 0) m[i, j] = 0;
        }

        /// <summary>
        /// Convolutes the input matrix with the given 3x3 kernel
        /// </summary>
        /// <param name="m">The input matrix</param>
        /// <param name="kernel">The 3x3 convolution kernel to use</param>
        [Pure]
        [NotNull]
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

        #endregion

        #region Misc

        /// <summary>
        /// Performs the multiplication between a vector and a matrix
        /// </summary>
        /// <param name="v">The input vector</param>
        /// <param name="m">The matrix to multiply</param>
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Multiply([NotNull] double[] v, [NotNull] double[,] m)
        {
            // Initialize the parameters and the result vector
            int w = m.GetLength(1);
            double[] result = new double[w];
            unsafe
            {
                // Get the pointers and iterate fo each column
                fixed (double* pm = result, p1 = v, p2 = m)
                {
                    for (int j = 0; j < w; j++)
                    {
                        // Perform the multiplication
                        int j2 = j;
                        double res = 0;
                        for (int k = 0; k < v.Length; k++, j2 += w)
                        {
                            res += p1[k] * p2[j2];
                        }
                        pm[j] = res;
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Transposes the input matrix
        /// </summary>
        /// <param name="m">The matrix to transpose</param>
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Transpose([NotNull] double[,] m)
        {
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[w, h];
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    result[j, i] = m[i, j];
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function has been applied
        /// </summary>
        /// <param name="v">The input to process</param>
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Sigmoid([NotNull] double[] v)
        {
            double[] result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
                result[i] = 1 / (1 + Math.Exp(-v[i]));
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function primed has been applied
        /// </summary>
        /// <param name="v">The input to process</param>
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] SigmoidPrime([NotNull] double[] v)
        {
            double[] result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                double
                    exp = Math.Exp(-v[i]),
                    sum = 1 + exp,
                    square = sum * sum,
                    div = exp / square;
                result[i] = div;
            }
            return result;
        }

        #endregion
    }
}
