using System;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.Networks.Layers.Initialization;

namespace NeuralNetworkNET.Networks.Layers.Cpu
{
    /// <summary>
    /// A fully connected (dense) network layer
    /// </summary>
    internal class FullyConnectedLayer : WeightedLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.FullyConnected;

        public FullyConnectedLayer(in TensorInfo input, int neurons, ActivationType activation, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode)
            : base(input, TensorInfo.Linear(neurons),
                  WeightsProvider.NewFullyConnectedWeights(input, neurons, weightsMode),
                  WeightsProvider.NewBiases(neurons, biasMode), activation) { }

        public FullyConnectedLayer(in TensorInfo input, int neurons, [NotNull] float[] weights, [NotNull] float[] biases, ActivationType activation)
            : base(input, TensorInfo.Linear(neurons), weights, biases, activation)
        {
            if (neurons != biases.Length)
                throw new ArgumentException("The biases vector must have the same size as the number of output neurons");
        }

        #region Implementation

        /// <inheritdoc/>
        public override unsafe void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            fixed (float* pw = Weights, pb = Biases)
            {
                Tensor.Reshape(pw, InputInfo.Size, OutputInfo.Size, out Tensor w);
                Tensor.Reshape(pb, 1, Biases.Length, out Tensor b);
                Tensor.New(x.Entities, OutputInfo.Size, out z);
                CpuDnn.FullyConnectedForward(x, w, b, z);
                Tensor.New(z.Entities, z.Length, out a);
                CpuDnn.ActivationForward(z, ActivationFunctions.Activation, a);
            }
        }

        /// <inheritdoc/>
        public override unsafe void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            // Backpropagation
            Tensor.Like(dy, out Tensor dy_copy);
            CpuDnn.ActivationBackward(y, dy, ActivationFunctions.ActivationPrime, dy_copy);
            if (!dx.IsNull) // Stop the error backpropagation if needed
            {
                fixed (float* pw = Weights)
                {
                    Tensor.Reshape(pw, InputInfo.Size, OutputInfo.Size, out Tensor w);
                    CpuDnn.FullyConnectedBackwardData(w, dy_copy, dx);
                }
            }

            // Gradient
            Tensor.New(InputInfo.Size, OutputInfo.Size, out Tensor dw);
            CpuDnn.FullyConnectedBackwardFilter(x, dy_copy, dw);
            dw.Reshape(1, dw.Size, out dJdw); // Flatten the result
            Tensor.New(1, Biases.Length, out dJdb);
            CpuDnn.FullyConnectedBackwardBias(dy_copy, dJdb);
            dy_copy.Free();
        }

        #endregion

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new FullyConnectedLayer(InputInfo, OutputInfo.Size, Weights.AsSpan().ToArray(), Biases.AsSpan().ToArray(), ActivationType);

        /// <summary>
        /// Tries to deserialize a new <see cref="FullyConnectedLayer"/> from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public static INetworkLayer Deserialize([NotNull] Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output)) return null;
            if (!stream.TryRead(out ActivationType activation)) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            return new FullyConnectedLayer(input, output.Size, weights, biases, activation);
        }
    }
}
