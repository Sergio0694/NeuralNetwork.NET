using System;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Cpu;

namespace NeuralNetworkNET.Networks.Layers.Cuda
{
    /// <summary>
    /// A cuDNN-powered fully connected layer
    /// </summary>
    internal class CuDnnFullyConnectedLayer : FullyConnectedLayer
    {
        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = CuDnnService.Instance;

        public CuDnnFullyConnectedLayer(in TensorInfo input, int neurons, ActivationType activation, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode) 
            : base(input, neurons, activation, weightsMode, biasMode) { }

        public CuDnnFullyConnectedLayer(in TensorInfo input, int neurons, [NotNull] float[] weights, [NotNull] float[] biases, ActivationType activation) 
            : base(input, neurons, weights, biases, activation) { }

        #region Implementation

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                w_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * OutputInfo.Size),
                b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
            {
                DnnInstance.FullyConnectedForward(x.Entities, x.Length, OutputInfo.Size, x_gpu.Ptr, w_gpu.Ptr, b_gpu.Ptr, y_gpu.Ptr);
                y_gpu.CopyToHost(x.Entities, OutputInfo.Size, out z);
                DnnInstance.ActivationForward(z.Entities, z.Length, y_gpu.Ptr, y_gpu.Ptr, ActivationFunctions.Activation);
                y_gpu.CopyToHost(z.Entities, z.Length, out a);
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            using (DeviceMemory<float>
                dy_gpu = DnnInstance.Gpu.AllocateDevice(dy),
                w_gpu = DnnInstance.Gpu.AllocateDevice(Weights), // Shared for the weights and dJdw, for better efficiency
                y_gpu = DnnInstance.Gpu.AllocateDevice(y),
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                dJdb_gpu = DnnInstance.Gpu.AllocateDevice<float>(Biases.Length))
            {
                // Backpropagation
                DnnInstance.ActivationBackward(y.Entities, y.Length, y_gpu.Ptr, dy_gpu.Ptr, ActivationFunctions.ActivationPrime, dy_gpu.Ptr);
                if (!dx.IsNull)
                {
                    using (DeviceMemory<float> dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(dx.Size))
                    {
                        DnnInstance.FullyConnectedBackwardData(y.Entities, InputInfo.Size, OutputInfo.Size, dy_gpu.Ptr, w_gpu.Ptr, dx_gpu.Ptr);
                        dx_gpu.CopyTo(dx);
                    }
                }

                // Gradient
                DnnInstance.FullyConnectedBackwardFilter(x.Entities, x.Length, dy.Length, x_gpu.Ptr, dy_gpu.Ptr, w_gpu.Ptr);
                w_gpu.CopyToHost(1, Weights.Length, out dJdw);
                DnnInstance.FullyConnectedBackwardBias(dy.Entities, dy.Length, dy_gpu.Ptr, dJdb_gpu.Ptr);
                dJdb_gpu.CopyToHost(1, Biases.Length, out dJdb);
            }
        }

        #endregion

        /// <summary>
        /// Tries to deserialize a new <see cref="CuDnnFullyConnectedLayer"/> from the input <see cref="System.IO.Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="System.IO.Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public new static INetworkLayer Deserialize([NotNull] System.IO.Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output)) return null;
            if (!stream.TryRead(out ActivationType activation)) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            return new CuDnnFullyConnectedLayer(input, output.Size, weights, biases, activation);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnFullyConnectedLayer(InputInfo, OutputInfo.Size, Weights.AsSpan().ToArray(), Biases.AsSpan().ToArray(), ActivationType);
    }
}
