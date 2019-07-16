using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs.Info;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Network.Initialization;
using NeuralNetworkDotNet.Network.Nodes.Unary.Abstract;

namespace NeuralNetworkDotNet.Network.Nodes.Unary
{
    /// <summary>
    /// A convolutional node
    /// </summary>
    internal sealed class ConvolutionalNode : WeightedUnaryNodeBase
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

        public ConvolutionalNode([NotNull] Node input, ConvolutionInfo operation, (int X, int Y) kernelSize, int kernels, BiasInitializationMode biasMode) : base(
            input,
            operation.GetOutputShape(input.Shape, kernelSize, kernels),
            WeightsProvider.NewConvolutionalKernels(input.Shape.C, kernelSize.X, kernelSize.Y, kernels),
            WeightsProvider.NewBiases(kernels, biasMode))
        {
            _OperationInfo = operation;
        }

        public ConvolutionalNode([NotNull] Node input, ConvolutionInfo operation, [NotNull] Tensor weights, [NotNull] Tensor biases)
            : base(input, operation.GetOutputShape(input.Shape, (weights.Shape.H, weights.Shape.W), weights.Shape.N), weights, biases)
        {
            _OperationInfo = operation;
        }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x)
        {
            var y = Tensor.New(x.Shape.N, Shape.C, Shape.H, Shape.W);
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
        public override bool Equals(Node other)
        {
            if (!base.Equals(other)) return false;

            return other is ConvolutionalNode node &&
                   OperationInfo.Equals(node.OperationInfo);
        }
    }
}
