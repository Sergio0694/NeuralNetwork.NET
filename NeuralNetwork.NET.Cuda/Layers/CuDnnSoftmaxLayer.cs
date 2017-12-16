using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
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

        #endregion

        public CuDnnSoftmaxLayer(int inputs, int outputs) : base(inputs, outputs) { }

        public CuDnnSoftmaxLayer([NotNull] float[,] weights, [NotNull] float[] biases) : base(weights, biases) { }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            MatrixGpuExtensions.MultiplyWithSum(x, Weights, Biases, out z);
            Gpu gpu = Gpu.Default;
            Dnn dnn = Dnn.Get(gpu);
            SoftmaxInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, Outputs, 1, 1);
            using (DeviceMemory<float>
                z_gpu = gpu.AllocateDevice(z),
                y_gpu = gpu.AllocateDevice<float>(z.Size))
            {
                dnn.SoftmaxForward(SoftmaxAlgorithm.FAST, SoftmaxMode.INSTANCE, 1, SoftmaxInfo, z_gpu.Ptr, 0, SoftmaxInfo, y_gpu.Ptr);
                y_gpu.CopyToHost(x.Entities, Outputs, out a);
            }
        }
    }
}