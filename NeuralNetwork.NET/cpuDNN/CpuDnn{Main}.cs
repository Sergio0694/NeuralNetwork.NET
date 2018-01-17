using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;

namespace NeuralNetworkNET.cpuDNN
{
    /// <summary>
    /// A static class that contains primitives to implement a CNN running on CPU
    /// </summary>
    public static partial class CpuDnn
    {
        #region Activation

        /// <summary>
        /// Executes the input activation function on the target <see cref="Tensor"/>
        /// </summary>
        /// <param name="x">The layer input <see cref="Tensor"/></param>
        /// <param name="f">The activation function to apply to the input</param>
        /// <param name="y">The output <see cref="Tensor"/> - it can be the same as the input</param>
        public static unsafe void ActivationForward(in Tensor x, [NotNull] ActivationFunction f, in Tensor y)
        {
            // Setup
            int n = x.Entities, l = x.Length;
            if (!y.MatchShape(x)) throw new ArgumentException("The target tensor must have the same shape as the input");
            float* py = y, px = x;

            // Execute the activation in parallel
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                {
                    int target = offset + j;
                    py[target] = f(px[target]);
                }
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Performs the softmax activation on the input <see cref="Tensor"/> and applies the output normalization
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/></param>
        /// <param name="y">The output <see cref="Tensor"/></param>
        public static unsafe void SoftmaxForward(in Tensor x, in Tensor y)
        {
            // Setup
            if (!x.MatchShape(y)) throw new ArgumentException("The input tensor doesn't have the same shape as the output tensor");
            int n = x.Entities, l = x.Length;
            Tensor.New(1, n, out Tensor partials);
            float* pp = partials, px = x, py = y;

            // Activation
            void ActivationWithAggregate(int i)
            {
                int offset = i * l;
                float sum = 0;
                for (int j = 0; j < l; j++)
                {
                    int target = offset + j;
                    float value = ActivationFunctions.Softmax(px[target]);
                    py[target] = value;
                    sum += value;
                }
                pp[i] = sum;
            }
            Parallel.For(0, n, ActivationWithAggregate).AssertCompleted();

            // Normalization of the tensor values
            void NormalizationKernel(int i)
            {
                int offset = i * l;
                float factor = pp[i];
                for (int j = 0; j < l; j++)
                    py[offset + j] /= factor;
            }
            Parallel.For(0, n, NormalizationKernel).AssertCompleted();
            partials.Free();
        }

        /// <summary>
        /// Executes the backward activation function on the target <see cref="Tensor"/>, with the given error delta
        /// </summary>
        /// <param name="x">The activity on the input layer</param>
        /// <param name="dy">The current error delta to backpropagate</param>
        /// <param name="f_">The derivative of the activation function used in the forward pass</param>
        /// <param name="dx">The resulting input error delta - it can be the same as the input <see cref="Tensor"/></param>
        public static unsafe void ActivationBackward(in Tensor x, in Tensor dy, [NotNull] ActivationFunction f_, in Tensor dx)
        {
            // Check
            if (!dy.MatchShape(x)) throw new ArgumentException("The input tensors must have the same shape", nameof(dy));
            if (!dx.MatchShape(x)) throw new ArgumentException("The output tensor must have the same shape as the input", nameof(dy));
            int
                n = x.Entities,
                l = x.Length;
            float* px = x, pdy = dy, pdx = dx;

            // Loop in parallel
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                {
                    int target = offset + j;
                    pdx[target] = f_(px[target]) * pdy[target];
                }
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }

        #endregion

        #region Fully connected

        /// <summary>
        /// Executes the forward pass on a fully connected layer
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to process</param>
        /// <param name="w">The layer weights</param>
        /// <param name="b">The layer biases</param>
        /// <param name="y">The output <see cref="Tensor"/> for the current layer</param>
        public static unsafe void FullyConnectedForward(in Tensor x, in Tensor w, in Tensor b, in Tensor y)
        {
            // Initialize the parameters and the result tensor
            if (x.Length != w.Entities) throw new ArgumentOutOfRangeException("Invalid tensors shapes");
            if (!b.MatchShape(1, w.Length)) throw new ArgumentException("Invalid biases shape", nameof(b));
            if (!y.MatchShape(x.Entities, w.Length)) throw new ArgumentException("The output tensor doesn't have the right shape", nameof(y));
            int
                h = x.Entities,
                l = x.Length,
                k = w.Length;
            float* pdy = y, px = x, pw = w, pb = b;

            // Execute the multiplication in parallel
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < k; j++)
                {
                    // Perform the multiplication
                    int start = j;
                    float res = 0;
                    for (int q = 0; q < l; q++, start += k)
                    {
                        res += px[offset + q] * pw[start];
                    }
                    pdy[i * k + j] = res + pb[j]; // Sum the input vector to each component
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Executes the backward pass on a fully connected layer
        /// </summary>
        /// <param name="x">The activity on the layer inputs</param>
        /// <param name="w">The layer weights</param>
        /// <param name="dy">The output error delta</param>
        /// <param name="f_">The derivative of the activation function used in the forward pass</param>
        /// <param name="dx">The resulting input error delta</param>
        public static unsafe void FullyConnectedBackwardData(in Tensor x, in Tensor w, in Tensor dy, [NotNull] ActivationFunction f_, in Tensor dx)
        {
            if (w.Entities != x.Length) throw new ArgumentException("The weights tensor doesn't have a valid shape", nameof(w));
            if (!x.MatchShape(dy.Entities, w.Entities)) throw new ArgumentException("The input tensor doesn't have the right shape", nameof(x));
            if (!dx.MatchShape(x)) throw new ArgumentException("The output tensor doesn't have the right shape", nameof(dx));
            Tensor.New(w.Length, w.Entities, out Tensor wt);
            CpuBlas.Transpose(w, wt);

            // Initialize the parameters and the result tensor
            int 
                h = dy.Entities,
                l = dy.Length,
                k = wt.Length;
            float* pdx = dx, px = x, pdy = dy, pwt = wt;

            // Execute the multiplication in parallel
            void Kernel(int i)
            {
                int i1 = i * l;
                for (int j = 0; j < k; j++)
                {
                    // Perform the multiplication
                    int i2 = j;
                    float res = 0;
                    for (int q = 0; q < l; q++, i2 += k)
                    {
                        res += pdy[i1 + q] * pwt[i2];
                    }

                    // res has now the tensor multiplication result for position [i, j]
                    int zIndex = i * k + j;
                    pdx[zIndex] = f_(px[zIndex]) * res;
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
            wt.Free();
        }

        /// <summary>
        /// Executes the backward pass on a fully connected layer to calculate the gradient with respect to the weights
        /// </summary>
        /// <param name="x">The layer inputs</param>
        /// <param name="dy">The layer output error delta</param>
        /// <param name="dw">The resulting weights gradient <see cref="Tensor"/></param>
        public static void FullyConnectedBackwardFilter(in Tensor x, in Tensor dy, in Tensor dw)
        {
            if (x.Entities != dy.Entities) throw new ArgumentException("The input tensor doesn't match the number of samples from the delta", nameof(x));
            Tensor.New(x.Length, x.Entities, out Tensor xt);
            CpuBlas.Transpose(x, xt);
            CpuBlas.Multiply(xt, dy, dw);
            xt.Free();
        }

        /// <summary>
        /// Executes the backward pass on a fully connected layer to calculate the gradient with respect to the biases
        /// </summary>
        /// <param name="dy">The layer output error delta</param>
        /// <param name="db">The resulting biases gradient <see cref="Tensor"/></param>
        public static unsafe void FullyConnectedBackwardBias(in Tensor dy, in Tensor db)
        {
            // Preliminary checks and declarations
            if (!db.MatchShape(1, dy.Length)) throw new ArgumentException("Invalid result tensor shape", nameof(db));
            int
                n = dy.Entities,
                l = dy.Length;
            float* pdy = dy, pdb = db;

            // Compress the tensor
            void Kernel(int j)
            {
                float sum = 0;
                for (int i = 0; i < n; i++)
                    sum += pdy[i * l + j];
                pdb[j] = sum;
            }
            Parallel.For(0, l, Kernel).AssertCompleted();
        }

        #endregion

        #region Depth concatenation

        /// <summary>
        /// Executes the forward pass on a depth stacking layer
        /// </summary>
        /// <param name="inputs">A <see cref="Span{T}"/> containing the input <see cref="Tensor"/> instances to stack</param>
        /// <param name="y">The output <see cref="Tensor"/></param>
        public static unsafe void DepthConcatenationForward(Span<Tensor> inputs, in Tensor y)
        {
            // Checks and offsets computation
            if (inputs.Length == 0) throw new ArgumentException("The inputs can't be empty", nameof(inputs));
            int
                n = y.Entities,
                count = 0;
            int* offsets = stackalloc int[inputs.Length];
            fixed (Tensor* p = &inputs.DangerousGetPinnableReference())
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    offsets[i] = count;
                    count += p[i].Length;
                    if (p[i].Entities != y.Entities) throw new ArgumentException("The number of samples must be the same for all tensors");
                }
            }
            if (y.Length != count) throw new ArgumentException("The target tensor doesn't have the right size", nameof(y));

            // Concatenate the tensors in parallel
            float* py = y;
            void Kernel(int i)
            {
                float*
                    psource = inputs[i],
                    ptarget = py + offsets[i];
                int l = inputs[i].Length;
                long bytes = sizeof(float) * l;
                for (int j = 0; j < n; j++, psource += l, ptarget += count)
                    Buffer.MemoryCopy(psource, ptarget, bytes, bytes);
            }
            Parallel.For(0, inputs.Length, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Executes the backward pass on a depth stacking layer
        /// </summary>
        /// <param name="dy">The input <see cref="Tensor"/> with the error delta to backpropagate</param>
        /// <param name="dx">A <see cref="Span{T}"/> with the target <see cref="Tensor"/> instances</param>
        public static unsafe void DepthConcatenationBackward(in Tensor dy, Span<Tensor> dx)
        {
            // Checks and offsets computation
            if (dx.Length == 0) throw new ArgumentException("The result span can't be empty", nameof(dx));
            int
                n = dy.Entities,
                count = 0;
            int* offsets = stackalloc int[dx.Length];
            fixed (Tensor* p = &dx.DangerousGetPinnableReference())
            {
                for (int i = 0; i < dx.Length; i++)
                {
                    offsets[i] = count;
                    count += p[i].Length;
                    if (p[i].Entities != dy.Entities) throw new ArgumentException("The number of samples must be the same for all tensors");
                }
            }
            if (dy.Length != count) throw new ArgumentException("The size of the output tensors doesn't match the size of the input tensor");

            // Backpropagate in parallel
            float* pdy = dy;
            void Kernel(int i)
            {
                float*
                    psource = pdy + offsets[i],
                    ptarget = dx[i];
                int l = dx[i].Length;
                long bytes = sizeof(float) * l;
                for (int j = 0; j < n; j++, psource += count, ptarget += l)
                    Buffer.MemoryCopy(psource, ptarget, bytes, bytes);
            }
            Parallel.For(0, dx.Length, Kernel);
        }

        #endregion

        #region Sum

        /// <summary>
        /// Executes the forward pass on a sum layer
        /// </summary>
        /// <param name="inputs">A <see cref="Span{T}"/> containing the input <see cref="Tensor"/> instances to sum</param>
        /// <param name="y">The output <see cref="Tensor"/></param>
        public static unsafe void SumForward(Span<Tensor> inputs, in Tensor y)
        {
            if (inputs.Length == 0) throw new ArgumentException("The inputs can't be empty", nameof(inputs));
            int
                count = inputs.Length,
                n = y.Entities,
                l = y.Length;
            fixed (Tensor * p = &inputs.DangerousGetPinnableReference())
            {
                // Initial checks
                float** ps = stackalloc float*[count];
                for (int i = 0; i < count; i++)
                {
                    if (p[i].Entities != n || p[i].Length != l)
                        throw new ArgumentException("The input tensors must have the same size as the output tensor", nameof(inputs));
                    ps[i] = p[i];
                }

                // Sum the tensors in parallel
                float* py = y;
                void Kernel(int i)
                {
                    int offset = i * l;
                    for (int j = 0; j < l; j++)
                    {
                        int target = offset + j;
                        py[target] = 0;
                        for (int z = 0; z < count; z++)
                            py[target] += ps[z][target];
                    }

                }
                Parallel.For(0, n, Kernel).AssertCompleted();
            }
        }

        #endregion
    }
}
