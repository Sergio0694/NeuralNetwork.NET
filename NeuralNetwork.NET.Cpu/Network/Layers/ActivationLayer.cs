using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Network.Activations;
using NeuralNetworkDotNet.Network.Activations.Delegates;
using NeuralNetworkDotNet.Network.Layers.Abstract;

namespace NeuralNetworkDotNet.Network.Layers
{
    /// <summary>
    /// An activation layer
    /// </summary>
    internal sealed class ActivationLayer : LayerBase
    {
        /// <summary>
        /// Gets the activation type used in the current layer
        /// </summary>
        public ActivationType ActivationType { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the network
        /// </summary>
        private readonly (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions;

        public ActivationLayer(Shape input, Shape output, ActivationType type) : base(input, output)
        {
            ActivationType = type;
            ActivationFunctions = ActivationFunctionProvider.GetActivations(type);
        }

        /// <inheritdoc/>
        public override Tensor Forward(in Tensor x)
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
        public override ILayer Clone() => new ActivationLayer(InputShape, OutputShape, ActivationType);
    }
}
