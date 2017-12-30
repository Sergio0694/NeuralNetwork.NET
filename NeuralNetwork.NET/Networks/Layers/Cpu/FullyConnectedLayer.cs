using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using System;
using System.IO;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A fully connected (dense) network layer
    /// </summary>
    internal class FullyConnectedLayer : WeightedLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.FullyConnected;

        public FullyConnectedLayer(in TensorInfo input, int neurons, ActivationFunctionType activation, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode)
            : base(input, TensorInfo.CreateLinear(neurons),
                  WeightsProvider.NewFullyConnectedWeights(input, neurons, weightsMode),
                  WeightsProvider.NewBiases(neurons, biasMode), activation) { }

        public FullyConnectedLayer(in TensorInfo input, int neurons, [NotNull] float[] weights, [NotNull] float[] biases, ActivationFunctionType activation)
            : base(input, TensorInfo.CreateLinear(neurons), weights, biases, activation)
        {
            if (neurons != biases.Length)
                throw new ArgumentException("The biases vector must have the same size as the number of output neurons");
        }

        /// <inheritdoc/>
        public override unsafe void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            fixed (float* pw = Weights)
            {
                Tensor.Reshape(pw, InputInfo.Size, OutputInfo.Size, out Tensor wTensor);
                x.MultiplyWithSum(wTensor, Biases, out z);
                z.Activation(ActivationFunctions.Activation, out a);
            }
        }

        /// <inheritdoc/>
        public override unsafe void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime)
        {
            fixed (float* pw = Weights)
            {
                Tensor.Reshape(pw, InputInfo.Size, OutputInfo.Size, out Tensor wTensor);
                wTensor.Transpose(out Tensor wt);
                z.InPlaceMultiplyAndHadamardProductWithActivationPrime(delta_1, wt, activationPrime);
                wt.Free();
            }
        }

        /// <inheritdoc/>
        public override void ComputeGradient(in Tensor a, in Tensor delta, out Tensor dJdw, out Tensor dJdb)
        {
            a.Transpose(out Tensor at);
            at.Multiply(delta, out Tensor dJdwM);
            dJdwM.Reshape(1, Weights.Length, out dJdw);
            at.Free();
            delta.CompressVertically(out dJdb);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new FullyConnectedLayer(InputInfo, OutputInfo.Size, Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType);

        /// <summary>
        /// Tries to deserialize a new <see cref="FullyConnectedLayer"/> from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public static INetworkLayer Deserialize([NotNull] Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output)) return null;
            if (!stream.TryRead(out ActivationFunctionType activation)) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            return new FullyConnectedLayer(input, output.Size, weights, biases, activation);
        }
    }
}
