using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Initialization;
using NeuralNetworkDotNet.Network.Nodes.Unary.Abstract;

namespace NeuralNetworkDotNet.Network.Nodes.Unary
{
    /// <summary>
    /// A fully connected (dense) network node
    /// </summary>
    internal sealed class FullyConnecteNode : WeightedUnaryNodeBase
    {
        public FullyConnecteNode([NotNull] INode input, int outputs, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode) : base(
            input, (input.Shape.CHW, outputs),
            WeightsProvider.NewFullyConnectedWeights(input.Shape.CHW, outputs, weightsMode),
            WeightsProvider.NewBiases(outputs, biasMode))
        {
            Guard.IsTrue(outputs >= 0, nameof(outputs), "The outputs must be a positive number");
        }

        public FullyConnecteNode([NotNull] INode input, int outputs, [NotNull] Tensor weights, [NotNull] Tensor biases)
            : base(input, (input.Shape.CHW, outputs), weights, biases)
        {
            Guard.IsTrue(outputs >= 0, nameof(outputs), "The outputs must be a positive number");
            Guard.IsTrue(weights.Shape == (input.Shape.CHW, 1, 1, outputs), "The input weights don't have the right shape");
            Guard.IsTrue(biases.Shape == (1, 1, 1, outputs), nameof(biases), "The biases don't have the right shape");
        }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x)
        {
            var y = Tensor.New(x.Shape.N, Shape.CHW);
            CpuDnn.FullyConnectedForward(x, Weights, Biases, y);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            var dx = Tensor.Like(x);
            CpuDnn.FullyConnectedBackwardData(Weights, dy, dx);

            return dx;
        }

        /// <inheritdoc/>
        public override void Gradient(Tensor x, Tensor dy, out Tensor dJdw, out Tensor dJdb)
        {
            dJdw = Tensor.Like(Weights);
            CpuDnn.FullyConnectedBackwardFilter(x, dy, dJdw);

            dJdb = Tensor.Like(Biases);
            CpuDnn.FullyConnectedBackwardBias(dy, dJdb);
        }
    }
}
