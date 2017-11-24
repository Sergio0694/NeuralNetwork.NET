using JetBrains.Annotations;
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
        public override int Inputs => InputVolume.Volume;

        /// <inheritdoc/>
        public override int Outputs => OutputVolume.Volume;

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

        public ConvolutionalLayer(VolumeInformation input, (int X, int Y) kernelSize, int kernels, ActivationFunctionType activation)
            : base(WeightsProvider.ConvolutionalKernels(kernelSize.X * kernelSize.Y * input.Depth, kernels), 
                  WeightsProvider.Biases(kernels), activation)
        {
            InputVolume = input;
            KernelVolume = (kernelSize.X, kernelSize.Y, input.Depth);
            OutputVolume = (input.Height - kernelSize.X + 1, input.Width - kernelSize.Y + 1, kernels);
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
            float[,] z = MatrixServiceProvider.ConvoluteForward(x, InputVolume, Weights, KernelVolume);
            z.InPlaceSum(OutputVolume.Depth, Biases);
            float[,] a = ActivationFunctionType == ActivationFunctionType.Identity
                ? z.BlockCopy()
                : z.Activation(ActivationFunctions.Activation);
            return (z, a);
        }

        /// <inheritdoc/>
        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime)
        {
            float[,]
                w180 = Weights.Rotate180(KernelVolume.Depth),
                delta = MatrixServiceProvider.ConvoluteBackwards(delta_1, OutputVolume, w180, KernelVolume);
            delta.InPlaceHadamardProductWithActivation(z, activationPrime);
            return delta;
        }

        /// <inheritdoc/>
        public override LayerGradient ComputeGradient(float[,] a, float[,] delta)
        {
            float[,]
                a180 = a.Rotate180(InputVolume.Depth),
                dJdw = MatrixServiceProvider.ConvoluteGradient(a180, InputVolume.Depth, delta, OutputVolume.Depth);
            float[] dJdb = delta.CompressVertically(OutputVolume.Depth);
            return new LayerGradient(dJdw, dJdb);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new ConvolutionalLayer(InputVolume, KernelVolume, OutputVolume, Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType);
    }
}
