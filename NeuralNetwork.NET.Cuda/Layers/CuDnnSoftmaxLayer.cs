using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.Cuda.Services;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Structs;
using NeuralNetworkNET.APIs.Misc;

namespace NeuralNetworkNET.Cuda.Layers
{
    /// <summary>
    /// A softmax output layer based on the cuDNN back-end
    /// </summary>
    internal sealed class CuDnnSoftmaxLayer : SoftmaxLayer
    {
        #region cuDNN fields

        // The NCHW tensor info for the layer softmax activation outputs
        [NotNull]
        private readonly TensorDescriptor SoftmaxInfo = new TensorDescriptor();

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = DnnService.Instance;

        #endregion

        public CuDnnSoftmaxLayer(in TensorInfo input, int outputs) : base(input, outputs) { }

        public CuDnnSoftmaxLayer([NotNull] float[,] weights, [NotNull] float[] biases) : base(weights, biases) { }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            using (DeviceMemory<float> z_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * OutputInfo.Size))
            {
                // Linear pass
                using (DeviceMemory2D<float>
                    x_gpu = DnnInstance.Gpu.AllocateDevice2D(x),
                    w_gpu = DnnInstance.Gpu.AllocateDevice(Weights))
                using (DeviceMemory<float> b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
                {
                    DnnInstance.FullyConnectedForward(x.Entities, x.Length, OutputInfo.Size, x_gpu.Ptr, x_gpu.PitchInElements.ToInt32(), w_gpu.Ptr, w_gpu.PitchInElements.ToInt32(), b_gpu.Ptr, z_gpu.Ptr, OutputInfo.Size);
                    z_gpu.CopyToHost(x.Entities, OutputInfo.Size, out z);
                }

                // Activation
                SoftmaxInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OutputInfo.Size, 1, 1);
                using (DeviceMemory<float> y_gpu = DnnInstance.Gpu.AllocateDevice<float>(z.Size))
                {
                    DnnInstance.SoftmaxForward(SoftmaxAlgorithm.FAST, SoftmaxMode.INSTANCE, 1, SoftmaxInfo, z_gpu.Ptr, 0, SoftmaxInfo, y_gpu.Ptr);
                    y_gpu.CopyToHost(x.Entities, OutputInfo.Size, out a);
                }
            }
        }
    }
}