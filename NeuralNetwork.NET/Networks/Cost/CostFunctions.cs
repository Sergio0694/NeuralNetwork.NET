using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
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
        public static unsafe float QuadraticCost(in Tensor yHat, in Tensor y)
        {
            // Detect the size of the inputs
            int h = yHat.Entities, w = yHat.Length;
            if (h != y.Entities || w != y.Length) throw new ArgumentException("The two matrices must have the same size");

            // Calculate the cost (half the squared difference)
            Tensor.New(h, 1, out Tensor v);

            // Kernel to compute the partial sum
            float* pv = v, pyHat = yHat, py = y;
            void Kernel(int i)
            {
                int offset = i * w;
                float sum = 0;
                for (int j = 0; j < w; j++)
                {
                    int target = offset + j;
                    float
                        delta = pyHat[target] - py[target],
                        square = delta * delta;
                    sum += square;
                }
                pv[i] = sum;
            }
            Parallel.For(0, h, Kernel).AssertCompleted();

            // Sum the partial costs
            float cost = 0;
            for (int i = 0; i < h; i++) cost += pv[i];
            v.Free();
            return cost / 2;
        }

        /// <summary>
        /// Calculates the cross-entropy cost for a given feedforward result
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        [Pure]
        public static unsafe float CrossEntropyCost(in Tensor yHat, in Tensor y)
        {
            // Detect the size of the inputs
            int h = yHat.Entities, w = yHat.Length;
            if (h != y.Entities || w != y.Length) throw new ArgumentException("The two matrices must have the same size");

            // Calculates the components for each training sample
            Tensor.New(h, 1, out Tensor v);

            // Function to calculate cost for each sample
            float* pyHat = yHat, py = y, pv = v;
            void Kernel(int i)
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
            Parallel.For(0, h, Kernel).AssertCompleted();

            // Sum the partial results and normalize
            float cost = 0;
            for (int i = 0; i < h; i++) cost += pv[i];
            v.Free();
            return -cost / h;
        }

        /// <summary>
        /// Calculates the log-likelyhood cost for the given outputs and expected results
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        [Pure]
        public static unsafe float LogLikelyhoodCost(in Tensor yHat, in Tensor y)
        {
            // Detect the size of the inputs
            int h = yHat.Entities, w = yHat.Length;
            if (h != y.Entities || w != y.Length) throw new ArgumentException("The two matrices must have the same size");

            // Calculates the components for each training sample
            Tensor.New(h, 1, out Tensor v);

            // Kernel to compute the partial sum
            float* pv = v, pyHat = yHat, py = y;
            void Kernel(int i)
            {
                int
                    offset = i * w,
                    iy = new Span<float>(py + offset, w).Argmax();
                pv[i] = -(float)Math.Log(pyHat[offset + iy]);
            }
            Parallel.For(0, h, Kernel).AssertCompleted();

            // Sum the partial costs
            float cost = 0;
            for (int i = 0; i < h; i++) cost += pv[i];
            v.Free();
            return cost;
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
        /// <param name="dx">The backpropagated error</param>
        public static unsafe void QuadraticCostPrime(in Tensor yHat, in Tensor y, in Tensor z, ActivationFunction activationPrime, in Tensor dx)
        {
            // Detect the size of the inputs
            int h = yHat.Entities, w = yHat.Length;
            if (h != y.Entities || w != y.Length) throw new ArgumentException("The two tensors must have the same shape");
            if (h != dx.Entities || w != dx.Length) throw new ArgumentException("The output tensor must have the same shape as the inputs");

            // Calculate (yHat - y) * activation'(z)
            float* pyHat = yHat, py = y, pz = z, pdx = dx;
            void Kernel(int i)
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
                    pdx[index] = hProduct;
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
        /// <param name="dx">The backpropagated error</param>
        public static void CrossEntropyCostPrime(in Tensor yHat, in Tensor y, in Tensor z, ActivationFunction activationPrime, in Tensor dx)
        {
            // Detect the size of the inputs
            int h = yHat.Entities, w = yHat.Length;
            if (h != y.Entities || w != y.Length) throw new ArgumentException("The two tensors must have the same shape");
            if (h != dx.Entities || w != dx.Length) throw new ArgumentException("The output tensor must have the same shape as the inputs");

            // Calculate (yHat - y)
            CpuBlas.Subtract(yHat, y, dx);
        }

        #endregion
    }
}
