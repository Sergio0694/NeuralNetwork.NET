using System;
using System.IO;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.Networks.Layers.Initialization;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Layers.Cpu
{
    /// <summary>
    /// A convolutional layer, used in a CNN network
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal class ConvolutionalLayer : WeightedLayerBase
    {
        #region Parameters

        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.Convolutional;

        [JsonProperty(nameof(OperationInfo), Order = 6)]
        private readonly ConvolutionInfo _OperationInfo;

        /// <summary>
        /// Gets the info on the convolution operation performed by the layer
        /// </summary>    
        public ref readonly ConvolutionInfo OperationInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _OperationInfo;
        }

        [JsonProperty(nameof(KernelInfo), Order = 7)]
        public readonly TensorInfo _KernelInfo;

        /// <summary>
        /// Gets the <see cref="TensorInfo"/> associated with each kernel in the layer
        /// </summary>

        public ref readonly TensorInfo KernelInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _KernelInfo;
        }

        /// <summary>
        /// Gets the number of kernels in the current layer
        /// </summary>
        [JsonProperty(nameof(Kernels), Order = 8)]
        public int Kernels => OutputInfo.Channels;

        #endregion

        public ConvolutionalLayer(in TensorInfo input, in ConvolutionInfo operation, (int X, int Y) kernelSize, int kernels, ActivationType activation, BiasInitializationMode biasMode)
            : base(input, operation.GetForwardOutputTensorInfo(input, kernelSize, kernels),
                  WeightsProvider.NewConvolutionalKernels(input, kernelSize.X, kernelSize.Y, kernels),
                  WeightsProvider.NewBiases(kernels, biasMode), activation)
        {
            _OperationInfo = operation;
            _KernelInfo = new TensorInfo(kernelSize.X, kernelSize.Y, input.Channels);
        }

        public ConvolutionalLayer(
            in TensorInfo input, in ConvolutionInfo operation, in TensorInfo kernels, in TensorInfo output,
            [NotNull] float[] weights, [NotNull] float[] biases, ActivationType activation)
            : base(input, output, weights, biases, activation)
        {
            _OperationInfo = operation;
            _KernelInfo = kernels;
        }

        #region Implementation

        /// <inheritdoc/>
        public override unsafe void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            fixed (float* pw = Weights, pb = Biases)
            {
                Tensor.Reshape(pw, OutputInfo.Channels, KernelInfo.Size, out Tensor w);
                Tensor.Reshape(pb, 1, Biases.Length, out Tensor b);
                Tensor.New(x.Entities, OutputInfo.Size, out z);
                CpuDnn.ConvolutionForward(x, InputInfo, w, KernelInfo, b, z);
                Tensor.New(z.Entities, z.Length, out a);
                if (ActivationType == ActivationType.Identity) a.Overwrite(z);
                else CpuDnn.ActivationForward(z, ActivationFunctions.Activation, a);
            }
        }

        /// <inheritdoc/>
        public override unsafe void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            // Backpropagation
            Tensor.Like(dy, out Tensor dy_copy);
            CpuDnn.ActivationBackward(y, dy, ActivationFunctions.ActivationPrime, dy_copy);
            if (!dx.IsNull)
            {
                fixed (float* pw = Weights)
                {
                    Tensor.Reshape(pw, OutputInfo.Channels, KernelInfo.Size, out Tensor wTensor);
                    CpuDnn.ConvolutionBackwardData(dy_copy, OutputInfo, wTensor, KernelInfo, dx, InputInfo);
                }
            }

            // Gradient
            Tensor.New(OutputInfo.Channels, KernelInfo.Size, out Tensor dw);
            CpuDnn.ConvolutionBackwardFilter(x, InputInfo, dy_copy, OutputInfo, dw, KernelInfo);
            dw.Reshape(1, Weights.Length, out dJdw);
            Tensor.New(1, Biases.Length, out dJdb);
            CpuDnn.ConvolutionBackwardBias(dy_copy, OutputInfo, dJdb);
            dy_copy.Free();
        }

        #endregion

        #region Misc

        /// <inheritdoc/>
        public override bool Equals(INetworkLayer other)
        {
            if (!base.Equals(other)) return false;
            if (!(other is ConvolutionalLayer convolutional)) return false;
            return convolutional.OperationInfo.Equals(OperationInfo) && convolutional.KernelInfo.Equals(KernelInfo);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new ConvolutionalLayer(InputInfo, OperationInfo, KernelInfo, OutputInfo, Weights.AsSpan().ToArray(), Biases.AsSpan().ToArray(), ActivationType);

        /// <inheritdoc/>
        public override void Serialize(Stream stream)
        {
            base.Serialize(stream);
            stream.Write(OperationInfo);
            stream.Write(KernelInfo);
        }

        /// <summary>
        /// Tries to deserialize a new <see cref="ConvolutionalLayer"/> from the input <see cref="Stream"/>
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
            if (!stream.TryRead(out ConvolutionInfo operation) && operation.Equals(ConvolutionInfo.Default)) return null;
            if (!stream.TryRead(out TensorInfo kernels)) return null;
            return new ConvolutionalLayer(input, operation, kernels, output, weights, biases, activation);
        }

        #endregion
    }
}
