using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Layers.Cpu;

namespace NeuralNetworkNET.Networks.Layers.Cuda
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
        private readonly Dnn DnnInstance = CuDnnService.Instance;

        #endregion

        public CuDnnSoftmaxLayer(in TensorInfo input, int outputs, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode) : base(input, outputs, weightsMode, biasMode) { }

        public CuDnnSoftmaxLayer(in TensorInfo input, int outputs, [NotNull] float[] weights, [NotNull] float[] biases) : base(input, outputs, weights, biases) { }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            using (DeviceMemory<float> z_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * OutputInfo.Size))
            {
                // Linear pass
                using (DeviceMemory<float>
                    x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                    w_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                    b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
                {
                    DnnInstance.FullyConnectedForward(x.Entities, x.Length, OutputInfo.Size, x_gpu.Ptr, w_gpu.Ptr, b_gpu.Ptr, z_gpu.Ptr);
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

        /// <summary>
        /// Tries to deserialize a new <see cref="CuDnnSoftmaxLayer"/> from the input <see cref="System.IO.Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public new static INetworkLayer Deserialize([NotNull] System.IO.Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output)) return null;
            if (!stream.TryRead(out ActivationFunctionType activation) && activation == ActivationFunctionType.Softmax) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            if (!stream.TryRead(out CostFunctionType cost) && cost == CostFunctionType.LogLikelyhood) return null;
            return new CuDnnSoftmaxLayer(input, output.Size, weights, biases);
        }
    }
}