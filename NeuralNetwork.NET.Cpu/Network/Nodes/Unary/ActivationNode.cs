using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Network.Activations;
using NeuralNetworkDotNet.Network.Activations.Delegates;
using NeuralNetworkDotNet.Network.Nodes.Abstract;
using NeuralNetworkDotNet.Network.Nodes.Enums;

namespace NeuralNetworkDotNet.Network.Nodes.Unary
{
    /// <summary>
    /// An activation node, that applies a specific activation function to its inputs
    /// </summary>
    internal class ActivationNode : UnaryNodeBase
    {
        /// <inheritdoc/>
        public override NodeType Type => NodeType.Activation;

        /// <summary>
        /// Gets the activation type used in the current layer
        /// </summary>
        public ActivationType ActivationType { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the network
        /// </summary>
        protected readonly (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions;

        public ActivationNode([NotNull] Node input, ActivationType type) : base(input, input.Shape)
        {
            ActivationType = type;
            ActivationFunctions = ActivationFunctionProvider.GetActivations(type);
        }

        /// <inheritdoc/>

        public override Tensor Forward(Tensor x)
        {
            var y = Tensor.Like(x);
            CpuDnn.ActivationForward(x, ActivationFunctions.Activation, y);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            var dx = Tensor.Like(x);
            CpuDnn.ActivationBackward(y, dy, ActivationFunctions.ActivationPrime, dx);

            return dx;
        }

        /// <inheritdoc/>
        public override bool Equals(Node other)
        {
            if (!base.Equals(other)) return false;

            return other is ActivationNode node &&
                   ActivationType.Equals(node.ActivationType);
        }
    }
}
