using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Layers.Abstract;
using System;

namespace NeuralNetworkNET.Networks.Layers.Cuda
{
    /// <summary>
    /// A softmax output layer based on the cuDNN back-end
    /// </summary>
    internal sealed class CuDnnSoftmaxLayer : OutputLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.Softmax;

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

        public CuDnnSoftmaxLayer(in TensorInfo input, int outputs, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode)
            : base(input, outputs, ActivationType.Softmax, CostFunctionType.LogLikelyhood, weightsMode, biasMode) { }

        public CuDnnSoftmaxLayer(in TensorInfo input, int outputs, [NotNull] float[] weights, [NotNull] float[] biases)
            : base(input, outputs, weights, biases, ActivationType.Softmax, CostFunctionType.LogLikelyhood) { }

        #region Implementation

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

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor yHat, in Tensor y, in Tensor z, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            using (DeviceMemory<float>
                yHat_gpu = DnnInstance.Gpu.AllocateDevice(yHat),
                dy_gpu = DnnInstance.Gpu.AllocateDevice(y),
                w_gpu = DnnInstance.Gpu.AllocateDevice(Weights), // Shared for the weights and dJdw, for better efficiency
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Size),
                dJdb_gpu = DnnInstance.Gpu.AllocateDevice<float>(Biases.Length))
            {
                // The derivative is just yHat - y
                DnnInstance.AddTensor(1, SoftmaxInfo, yHat_gpu.Ptr, -1, SoftmaxInfo, dy_gpu.Ptr);
                DnnInstance.FullyConnectedBackwardData(y.Entities, InputInfo.Size, OutputInfo.Size, dy_gpu.Ptr, w_gpu.Ptr, dx_gpu.Ptr);
                dx_gpu.CopyTo(dx);

                // Gradient
                DnnInstance.FullyConnectedBackwardFilter(x.Entities, x.Length, y.Length, x_gpu.Ptr, dy_gpu.Ptr, w_gpu.Ptr);
                w_gpu.CopyToHost(1, Weights.Length, out dJdw);
                DnnInstance.FullyConnectedBackwardBias(y.Entities, y.Length, dy_gpu.Ptr, dJdb_gpu.Ptr); // Doing this on CPU is generally faster than launching the kernels
                dJdb_gpu.CopyToHost(1, Biases.Length, out dJdb);
            }
        }

        #endregion

        /// <summary>
        /// Tries to deserialize a new <see cref="CuDnnSoftmaxLayer"/> from the input <see cref="System.IO.Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public new static INetworkLayer Deserialize([NotNull] System.IO.Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output)) return null;
            if (!stream.TryRead(out ActivationType activation) && activation == ActivationType.Softmax) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            if (!stream.TryRead(out CostFunctionType cost) && cost == CostFunctionType.LogLikelyhood) return null;
            return new CuDnnSoftmaxLayer(input, output.Size, weights, biases);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnSoftmaxLayer(InputInfo, OutputInfo.Size, Weights.AsSpan().Copy(), Biases.AsSpan().Copy());
    }
}