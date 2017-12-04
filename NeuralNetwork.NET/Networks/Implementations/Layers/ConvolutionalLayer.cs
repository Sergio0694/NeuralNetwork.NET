using JetBrains.Annotations;
using NeuralNetworkNET.DependencyInjection;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using NeuralNetworkNET.Structs;
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
            : base(WeightsProvider.ConvolutionalKernels(input.Depth, kernelSize.X, kernelSize.Y, kernels),
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
        public override unsafe void Forward(in FloatSpan2D x, out FloatSpan2D z, out FloatSpan2D a)
        {
            MatrixServiceProvider.ConvoluteForward(x, InputVolume, Weights, KernelVolume, Biases, out z);
            if (ActivationFunctionType == ActivationFunctionType.Identity) FloatSpan2D.From(z, z.Height, z.Width, out a);
            else MatrixServiceProvider.Activation(z, ActivationFunctions.Activation, out a);
        }

        /// <inheritdoc/>
        public override unsafe void Backpropagate(in FloatSpan2D delta_1, in FloatSpan2D z, ActivationFunction activationPrime)
        {
            fixed (float* pw = Weights)
            {
                FloatSpan2D.Fix(pw, Weights.GetLength(0), Weights.GetLength(1), out FloatSpan2D weights);
                weights.Rotate180(KernelVolume.Depth, out FloatSpan2D w180);
                MatrixServiceProvider.ConvoluteBackwards(delta_1, OutputVolume, w180, KernelVolume, out FloatSpan2D delta);
                w180.Free();
                z.InPlaceActivationAndHadamardProduct(delta, activationPrime);
                delta.Free();
            }
        }

        /// <inheritdoc/>
        public override void ComputeGradient(in FloatSpan2D a, in FloatSpan2D delta, out FloatSpan2D dJdw, out FloatSpan dJdb)
        {
            a.Rotate180(InputVolume.Depth, out FloatSpan2D a180);
            MatrixServiceProvider.ConvoluteGradient(a180, InputVolume, delta, OutputVolume, out dJdw);
            a180.Free();
            delta.CompressVertically(out dJdb);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new ConvolutionalLayer(InputVolume, KernelVolume, OutputVolume, Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType);
    }
}
