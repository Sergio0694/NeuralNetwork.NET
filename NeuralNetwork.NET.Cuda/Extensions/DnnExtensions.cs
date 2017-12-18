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
        #region Activation

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
        public static void ActivationForward([NotNull] this Dnn dnn, int n, int w, deviceptr<float> x, int ldx, deviceptr<float> y, int ldy, [NotNull] ActivationFunction f)
        {
            void Kernel(int i)
            {
                int
                    x_offset = i * ldx,
                    y_offset = i * ldy;
                for (int j = 0; j < w; j++)
                    y[y_offset + j] = f(x[x_offset + j]);
            }
            dnn.Gpu.For(0, n, Kernel);
        }

        /// <summary>
        /// Executes the backward activation function on the target memory area, with the given error delta
        /// </summary>
        /// <param name="dnn">The current <see cref="Dnn"/> instance being used</param>
        /// <param name="n">The number of samples in the input tensor</param>
        /// <param name="w">The size of each sample to process</param>
        /// <param name="z">A pointer to the input memory area</param>
        /// <param name="ldz">The main dimension of the input memory area (the pitch)</param>
        /// <param name="delta">The delta memory area</param>
        /// <param name="lddelta">The main dimension of the delta memory area</param>
        /// <param name="f">The activation function to use</param>
        public static void ActivationBackward([NotNull] this Dnn dnn, int n, int w, deviceptr<float> z, int ldz, deviceptr<float> delta, int lddelta, [NotNull] ActivationFunction f)
        {
            void Kernel(int i)
            {
                int
                    z_offset = i * ldz,
                    delta_offset = i * lddelta;
                for (int j = 0; j < w; j++)
                {
                    int z_position = z_offset + j;
                    z[z_position] = f(z[z_position]) * delta[delta_offset + j];
                }
            }
            dnn.Gpu.For(0, n, Kernel);
        }

        #endregion

        #region Fully connected

        /// <summary>
        /// Executes the forward pass on a fully connected layer
        /// </summary>
        /// <param name="dnn">The current <see cref="Dnn"/> instance being used</param>
        /// <param name="n">The number of samples in the input tensor</param>
        /// <param name="l">The size of each input sample to process</param>
        /// <param name="k">The number of output features for each sample</param>
        /// <param name="x">A pointer to the input memory area</param>
        /// <param name="ldx">The main dimension of the input memory area (the pitch)</param>
        /// <param name="w">A pointer to the layer weights</param>
        /// <param name="ldw">The main dimension of the weights memory area</param>
        /// <param name="b">A pointer to the network biases</param>
        /// <param name="y">A pointer to the output memory area</param>
        /// <param name="ldy">The main dimension of the output memory area</param>
        public static void FullyConnectedForward([NotNull] this Dnn dnn, int n, int l, int k, deviceptr<float> x, int ldx, deviceptr<float> w, int ldw, deviceptr<float> b, deviceptr<float> y, int ldy)
        {
            void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    i = index / k,
                    j = index % k;

                // Perform the multiplication
                float sum = 0;
                int pm1_offset = i * ldx;
                for (int z = 0; z < l; z++)
                {
                    sum += x[pm1_offset + z] * w[z * ldw + j];
                }
                y[i * ldy + j] = sum + b[j]; // Sum the input vector to each component
            }
            dnn.Gpu.For(0, n * k, Kernel);
        }

        /// <summary>
        /// Executes the backward pass on a fully connected layer
        /// </summary>
        /// <param name="dnn">The current <see cref="Dnn"/> instance being used</param>
        /// <param name="n">The number of samples in the input tensor</param>
        /// <param name="k">The number of output features for each sample</param>
        /// <param name="l">The size of each input error delta to process</param>
        /// <param name="z">A pointer to the activity on the previous layer</param>
        /// <param name="ldz">The main dimension of the activity memory area (the pitch)</param>
        /// <param name="dy">A pointer to the output error delta</param>
        /// <param name="lddy">The main dimension of the output delta memory area</param>
        /// <param name="w">A pointer to the layer weights</param>
        /// <param name="ldw">The main dimension of the weights memory area</param>
        /// <param name="f_">The derivative of the activation function of the previous layer</param>
        public static void FullyConnectedBackwardData([NotNull] this Dnn dnn, int n, int k, int l, deviceptr<float> z, int ldz, deviceptr<float> dy, int lddy, deviceptr<float> w, int ldw, [NotNull] ActivationFunction f_)
        {
            void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    i = index / k,
                    j = index % k;

                // Perform the multiplication (the second matrix is transposed while being processed)
                float sum = 0;
                int
                    dy_offset = i * lddy,
                    w_offset = j * ldw;
                for (int iter = 0; iter < l; iter++)
                {
                    sum += dy[dy_offset + iter] * w[w_offset + iter];
                }

                // Activation
                int z_offset = i * ldz + j;
                z[z_offset] = f_(z[z_offset]) * sum;
            }
            dnn.Gpu.For(0, n * k, Kernel);
        }

        #endregion
    }
}
