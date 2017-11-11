using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;

namespace NeuralNetworkNET.Networks.Cost
{
    /// <summary>
    /// A collection of cost functions available for the neural networks
    /// </summary>
    public static class CostFunctions
    {
        /// <summary>
        /// Calculates half the sum of the squared difference of each value pair in the two matrices
        /// </summary>
        /// <param name="yHat">The first matrix</param>
        /// <param name="y">The second matrix</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float QuadraticCost([NotNull] this float[,] yHat, [NotNull] float[,] y)
        {
            // Detect the size of the inputs
            int h = yHat.GetLength(0), w = yHat.GetLength(1);
            if (h != y.GetLength(0) || w != y.GetLength(1)) throw new ArgumentException("The two matrices must have the same size");

            // Calculate the cost (half the squared difference)
            float[] v = new float[h];

            // Kernel to compute the partial sum
            unsafe void Kernel(int i)
            {
                fixed (float* pv = v, pyHat = yHat, py = y)
                {
                    int offset = i * w;
                    for (int j = 0; j < w; j++)
                    {
                        int target = offset + j;
                        float
                            delta = pyHat[target] - py[target],
                            square = delta * delta;
                        pv[i] += square;
                    }
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();

            // Sum the partial costs
            float cost = 0;
            unsafe
            {
                fixed (float* pv = v)
                    for (int i = 0; i < h; i++) cost += pv[i];
            }
            return cost / 2;
        }

        /// <summary>
        /// Calculates the cross-entropy cost for a given feedforward result
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        public static float CrossEntropyCost([NotNull] this float[,] yHat, [NotNull] float[,] y)
        {
            // Detect the size of the inputs
            int h = yHat.GetLength(0), w = yHat.GetLength(1);
            if (h != y.GetLength(0) || w != y.GetLength(1)) throw new ArgumentException("The two matrices must have the same size");

            // Calculates the components for each training sample
            float[] v = new float[h];

            // Function to calculate cost for each sample
            unsafe void Kernel(int i)
            {
                fixed (float* pyHat = yHat, py = y, pv = v)
                {
                    int offset = i * w;
                    float sum = 0;
                    for (int j = 0; j < w; j++)
                    {
                        int target = offset + j;
                        float
                            yi = py[target],
                            yHati = pyHat[target],
                            left = yi * (float)Math.Log(yHati),
                            right = (1 - yi) * (float)Math.Log(1 - yHati),
                            partial = left + right;
                        switch (partial)
                        {
                            case float.NegativeInfinity:
                                sum += -float.MaxValue;
                                break;
                            case float.NaN:
                                break;
                            case float.PositiveInfinity:
                                throw new InvalidOperationException("Error calculating the cross-entropy cost");
                            default:
                                sum += partial;
                                break;
                        }
                    }
                    pv[i] = sum;
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();

            // Sum the partial results and normalize
            float cost = 0;
            unsafe
            {
                fixed (float* pv = v)
                    for (int i = 0; i < h; i++) cost += pv[i];
            }
            return -cost / h;
        }
    }
}
