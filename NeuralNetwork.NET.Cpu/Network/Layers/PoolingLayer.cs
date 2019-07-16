using System.Runtime.CompilerServices;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.APIs.Structs.Info;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Network.Layers.Abstract;

namespace NeuralNetworkDotNet.Network.Layers
{
    /// <summary>
    /// A pooling layer, with a 2x2 window and a stride of 2
    /// </summary>
    internal sealed class PoolingLayer : LayerBase
    {
        private readonly PoolingInfo _OperationInfo;

        /// <summary>
        /// Gets the info on the pooling operation performed by the layer
        /// </summary>
        public ref readonly PoolingInfo OperationInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _OperationInfo;
        }

        public PoolingLayer(Shape input, PoolingInfo operation) : base(input, operation.GetOutputShape(input))
        {
            _OperationInfo = operation;
        }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x)
        {
            var y = Tensor.New(x.Shape.N, OutputShape.C, OutputShape.H, OutputShape.W);
            CpuDnn.PoolingForward(x, y);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            var dx = Tensor.Like(x);
            CpuDnn.PoolingBackward(x, dy, dx);

            return dx;
        }

        /// <inheritdoc/>
        public override bool Equals(ILayer other)
        {
            if (!base.Equals(other)) return false;

            return other is PoolingLayer layer &&
                   OperationInfo.Equals(layer.OperationInfo);
        }

        /// <inheritdoc/>
        public override ILayer Clone() => new PoolingLayer(InputShape, OperationInfo);
    }
}