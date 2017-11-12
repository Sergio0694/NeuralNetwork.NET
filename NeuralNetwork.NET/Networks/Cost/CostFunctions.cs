using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations.Delegates;

namespace NeuralNetworkNET.Networks.Cost
{
    /// <summary>
    /// A collection of cost functions available for the neural networks
    /// </summary>
    public static class CostFunctions
    {
        #region Cost

        /// <summary>
        /// Calculates the quadratic cost for the given outputs and expected results
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float QuadraticCost([NotNull] float[,] yHat, [NotNull] float[,] y)
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
        public static float CrossEntropyCost([NotNull] float[,] yHat, [NotNull] float[,] y)
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

        #endregion

        #region Derivative

        /// <summary>
        /// Calculates the derivative of the quadratic cost function for the given outputs, expected results and activity
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        /// <param name="z">The activity on the last network layer</param>
        /// <param name="activationPrime">The activation pime function for the last network layer</param>
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void QuadraticCostPrime([NotNull] float[,] yHat, [NotNull] float[,] y, [NotNull] float[,] z, ActivationFunction activationPrime)
        {
            // Detect the size of the inputs
            int h = yHat.GetLength(0), w = yHat.GetLength(1);
            if (h != y.GetLength(0) || w != y.GetLength(1)) throw new ArgumentException("The two matrices must have the same size");

            // Calculate (yHat - y) * activation'(z)
            unsafe void Kernel(int i)
            {
                // Get the pointers and iterate fo each row
                fixed (float* pyHat = yHat, py = y, pz = z)
                {
                    // Save the index and iterate for each column
                    int offset = i * w;
                    for (int j = 0; j < w; j++)
                    {
                        int index = offset + j;
                        float
                            difference = pyHat[index] - py[index],
                            zPrime = activationPrime(pz[index]),
                            hProduct = difference * zPrime;
                        pyHat[index] = hProduct;
                    }
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Calculates the derivative cross-entropy cost for a given feedforward result
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        /// <param name="z">The activity on the last network layer</param>
        /// <param name="activationPrime">The activation pime function for the last network layer</param>
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void CrossEntropyCostPrime([NotNull] float[,] yHat, [NotNull] float[,] y, [NotNull] float[,] z, ActivationFunction activationPrime)
        {
            // Detect the size of the inputs
            int h = yHat.GetLength(0), w = yHat.GetLength(1);
            if (h != y.GetLength(0) || w != y.GetLength(1)) throw new ArgumentException("The two matrices must have the same size");

            // Calculate (yHat - y)
            yHat.Subtract(y);
        }

        #endregion
    }
}
