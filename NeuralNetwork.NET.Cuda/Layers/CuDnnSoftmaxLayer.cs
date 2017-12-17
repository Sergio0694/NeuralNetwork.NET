using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.Cuda.Services;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Structs;

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

        public CuDnnSoftmaxLayer(int inputs, int outputs) : base(inputs, outputs) { }

        public CuDnnSoftmaxLayer([NotNull] float[,] weights, [NotNull] float[] biases) : base(weights, biases) { }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            Blas.MultiplyWithSum(x, Weights, Biases, out z);
            SoftmaxInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, Outputs, 1, 1);
            using (DeviceMemory<float>
                z_gpu = DnnInstance.Gpu.AllocateDevice(z),
                y_gpu = DnnInstance.Gpu.AllocateDevice<float>(z.Size))
            {
                DnnInstance.SoftmaxForward(SoftmaxAlgorithm.FAST, SoftmaxMode.INSTANCE, 1, SoftmaxInfo, z_gpu.Ptr, 0, SoftmaxInfo, y_gpu.Ptr);
                y_gpu.CopyToHost(x.Entities, Outputs, out a);
            }
        }
    }
}