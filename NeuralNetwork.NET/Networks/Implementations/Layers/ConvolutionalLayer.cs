using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using NeuralNetworkNET.Networks.Implementations.Misc;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A convolutional layer, used in a CNN network
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal sealed class ConvolutionalLayer : WeightedLayerBase, INetworkLayer3D
    {
        #region Parameters

        /// <inheritdoc/>
        public override int Inputs => InputVolume.Size;

        /// <inheritdoc/>
        public override int Outputs => OutputVolume.Size;

        /// <inheritdoc/>
        [JsonProperty(nameof(InputVolume), Order = 4)]
        public VolumeInformation InputVolume { get; }

        /// <summary>
        /// Gets the <see cref="VolumeInformation"/> associated with each kernel in the layer
        /// </summary>
        [JsonProperty(nameof(KernelVolume), Order = 5)]
        public VolumeInformation KernelVolume { get; }

        /// <summary>
        /// Gets the number of kernels in the current layer
        /// </summary>
        [JsonProperty(nameof(Kernels), Order = 6)]
        public int Kernels => Weights.GetLength(0);

        /// <inheritdoc/>
        [JsonProperty(nameof(OutputVolume), Order = 7)]
        public VolumeInformation OutputVolume { get; }

        #endregion

        public ConvolutionalLayer(VolumeInformation input, int kernelAxis, int kernels, ActivationFunctionType activation)
            : base(WeightsProvider.ConvolutionalKernels(kernelAxis, input.Depth, kernels), 
                  WeightsProvider.Biases(kernels), activation)
        {
            InputVolume = input;
            KernelVolume = new VolumeInformation(kernelAxis, input.Depth);
            OutputVolume = new VolumeInformation(input.Axis - kernelAxis + 1, kernels);
        }

        public ConvolutionalLayer(VolumeInformation input, VolumeInformation kernels, VolumeInformation output,
            [NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation)
            : base(weights, biases, activation)
        {
            InputVolume = input;
            KernelVolume = kernels;
            OutputVolume = output;
        }

        /// <inheritdoc/>
        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            float[,]
                z = x.Convolute(InputVolume.Depth, Weights, InputVolume.Depth, ConvolutionMode.Forward),
                a = z.Activation(ActivationFunctions.Activation);
            return (z, a);
        }

        /// <inheritdoc/>
        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public override LayerGradient ComputeGradient(float[,] a, float[,] delta)
        {
            float[,]
                a180 = a.Rotate180(InputVolume.Depth),
                dJdw = a180.Convolute(InputVolume.Depth, delta, OutputVolume.Depth, ConvolutionMode.Gradient);
            float[] dJdb = delta.CompressVertically(OutputVolume.Depth);
            return new LayerGradient(dJdw, dJdb);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new ConvolutionalLayer(InputVolume, KernelVolume, OutputVolume, Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType);
    }
}
