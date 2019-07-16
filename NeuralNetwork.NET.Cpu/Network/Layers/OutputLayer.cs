using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Network.Cost;
using NeuralNetworkDotNet.Network.Cost.Delegates;

namespace NeuralNetworkDotNet.Network.Layers
{
    /// <summary>
    /// A custom <see cref="ActivationLayer"/> used as output in a graph, that also contains a specific cost function
    /// </summary>
    internal class OutputLayer : ActivationLayer
    {
        /// <summary>
        /// Gets the cost function for the current layer
        /// </summary>
        public CostFunctionType CostFunctionType { get; }

        /// <summary>
        /// Gets the cost function implementations used in the current layer
        /// </summary>
        public (CostFunction Cost, CostFunctionPrime CostPrime) CostFunctions { get; }

        public OutputLayer(Shape input, Shape output, ActivationType activation, CostFunctionType costFunctionType)
            : base(input, output, activation)
        {
            CostFunctionType = costFunctionType;
            CostFunctions = CostFunctionProvider.GetCostFunctions(costFunctionType);
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            var dx = Tensor.Like(x);
            CostFunctions.CostPrime(dy, y, x, ActivationFunctions.ActivationPrime, dx);

            return dx;
        }

        /// <inheritdoc/>
        public override bool Equals(ILayer other)
        {
            if (!base.Equals(other)) return false;

            return other is OutputLayer layer &&
                   CostFunctionType.Equals(layer.CostFunctionType);
        }

        /// <inheritdoc/>
        public override ILayer Clone() => new OutputLayer(InputShape, OutputShape, ActivationType, CostFunctionType);
    }
}
