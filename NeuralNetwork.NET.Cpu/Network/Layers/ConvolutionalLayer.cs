using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.APIs.Structs.Info;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Network.Initialization;
using NeuralNetworkDotNet.Network.Layers.Abstract;

namespace NeuralNetworkDotNet.Network.Layers
{
    /// <summary>
    /// A convolutional layer
    /// </summary>
    internal sealed class ConvolutionalLayer : WeightedLayerBase
    {
        private readonly ConvolutionInfo _OperationInfo;

        /// <summary>
        /// Gets the info on the convolution operation performed by the layer
        /// </summary>    
        public ref readonly ConvolutionInfo OperationInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _OperationInfo;
        }

        public ConvolutionalLayer(Shape input, ConvolutionInfo operation, (int X, int Y) kernelSize, int kernels, BiasInitializationMode biasMode) : base(
            input,
            operation.GetOutputShape(input, kernelSize, kernels),
            WeightsProvider.NewConvolutionalKernels(input.C, kernelSize.X, kernelSize.Y, kernels),
            WeightsProvider.NewBiases(kernels, biasMode))
        {
            _OperationInfo = operation;
        }

        public ConvolutionalLayer(Shape input, ConvolutionInfo operation, [NotNull] Tensor weights, [NotNull] Tensor biases)
            : base(input, operation.GetOutputShape(input, (weights.Shape.H, weights.Shape.W), weights.Shape.N), weights, biases)
        {
            _OperationInfo = operation;
        }

        /// <inheritdoc/>
        public override Tensor Forward(in Tensor x)
        {
            var y = Tensor.New(x.Shape.N, OutputShape.C, OutputShape.H, OutputShape.W);
            CpuDnn.ConvolutionForward(x, Weights, Biases, y);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            var dx = Tensor.Like(x);
            CpuDnn.ConvolutionBackwardData(dy, Weights, dx);

            return dx;
        }

        /// <inheritdoc/>
        public override void Gradient(Tensor x, Tensor dy, out Tensor dJdw, out Tensor dJdb)
        {
            dJdw = Tensor.Like(Weights);
            CpuDnn.ConvolutionBackwardFilter(x, dy, dJdw);

            dJdb = Tensor.Like(Biases);
            CpuDnn.ConvolutionBackwardBias(dy, dJdb);
        }

        /// <inheritdoc/>
        public override bool Equals(ILayer other)
        {
            if (!base.Equals(other)) return false;

            return other is ConvolutionalLayer layer &&
                   OperationInfo.Equals(layer.OperationInfo);
        }

        /// <inheritdoc/>
        public override ILayer Clone() => new ConvolutionalLayer(InputShape, OperationInfo, Weights.Clone(), Biases.Clone());
    }
}
