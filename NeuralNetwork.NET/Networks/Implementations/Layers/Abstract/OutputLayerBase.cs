using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Cost.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Structs;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations.Layers.Abstract
{
    /// <summary>
    /// The base class for all the output network layers
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal abstract class OutputLayerBase : FullyConnectedLayer
    {
        #region Fields and parameters

        /// <summary>
        /// Gets the cost function for the current layer
        /// </summary>
        [JsonProperty(nameof(CostFunctionType), Required = Required.Always, Order = 9)]
        public CostFunctionType CostFunctionType { get; }

        /// <summary>
        /// Gets the cost function implementations used in the current layer
        /// </summary>
        public (CostFunction Cost, CostFunctionPrime CostPrime) CostFunctions { get; }

        #endregion

        protected OutputLayerBase(int inputs, int outputs, ActivationFunctionType activation, CostFunctionType cost)
            : base(inputs, outputs, activation)
        {
            CostFunctionType = cost;
            CostFunctions = CostFunctionProvider.GetCostFunctions(cost);
        }

        protected OutputLayerBase([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation, CostFunctionType cost)
            : base(weights, biases, activation)
        {
            CostFunctionType = cost;
            CostFunctions = CostFunctionProvider.GetCostFunctions(cost);
        }

        /// <summary>
        /// Computes the output delta, with respect to the cost function of the network
        /// </summary>
        /// <param name="yHat">The estimated outputs for the network</param>
        /// <param name="y">The expected outputs for the used inputs</param>
        /// <param name="z">The activity on the output layer</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public float[,] Backpropagate(in FloatSpan2D yHat, in FloatSpan2D y, in FloatSpan2D z)
        {
            CostFunctions.CostPrime(yHat, y, z, ActivationFunctions.ActivationPrime);
            return yHat;
        }

        /// <summary>
        /// Calculates the output cost with respect to the cost function currently in use
        /// </summary>
        /// <param name="yHat">The estimated output for the network</param>
        /// <param name="y">The Expected outputs for the inputs used</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public float CalculateCost([NotNull] float[,] yHat, [NotNull] float[,] y) => CostFunctions.Cost(yHat, y);

        #region Equality check

        /// <inheritdoc/>
        public override bool Equals(INetworkLayer other)
        {
            if (!base.Equals(other)) return false;
            return other is OutputLayerBase layer &&
                   CostFunctionType == layer.CostFunctionType;
        }

        #endregion
    }
}
