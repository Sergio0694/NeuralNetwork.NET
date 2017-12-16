using Alea;
using Alea.cuDNN;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations.Delegates;

namespace NeuralNetworkNET.Cuda.Extensions
{
    /// <summary>
    /// A static class with some extensions for the <see cref="Dnn"/> class
    /// </summary>
    internal static class DnnExtensions
    {
        /// <summary>
        /// Executes the input activation function on the target memory area. The input and output pointers can be the same, if needed
        /// </summary>
        /// <param name="dnn">The current <see cref="Dnn"/> instance being used</param>
        /// <param name="n">The number of samples in the input tensor</param>
        /// <param name="w">The size of each sample to process</param>
        /// <param name="x">A pointer to the input memory area</param>
        /// <param name="ldx">The main dimension of the input memory area (the pitch)</param>
        /// <param name="y">The output memory area</param>
        /// <param name="ldy">The main dimension of the output memory area</param>
        /// <param name="f">The activation function to use</param>
        public static void Activation([NotNull] this Dnn dnn, int n, int w, deviceptr<float> x, int ldx, deviceptr<float> y, int ldy, [NotNull] ActivationFunction f)
        {
            // Wrapper
            void Kernel(int i)
            {
                int
                    x_offset = i * ldx,
                    y_offset = i * ldy;
                for (int j = 0; j < w; j++)
                    y[y_offset + j] = f(x[x_offset + j]);
            }

            // Execute the multiplication in parallel
            dnn.Gpu.For(0, n, Kernel);
        }
    }
}
