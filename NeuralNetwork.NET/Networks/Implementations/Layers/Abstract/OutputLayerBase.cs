using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Cost.Delegates;

namespace NeuralNetworkNET.Networks.Implementations.Layers.Abstract
{
    /// <summary>
    /// The base class for all the output network layers
    /// </summary>
    internal abstract class OutputLayerBase : FullyConnectedLayer
    {
        /// <summary>
        /// Gets the <see cref="CostFunction"/> used to evaluate the neural network
        /// </summary>
        [NotNull]
        private readonly CostFunction CostFunction;

        /// <summary>
        /// Gets the <see cref="CostFunctionPrime"/> used in the gradient descent algorithm
        /// </summary>
        [NotNull]
        private readonly CostFunctionPrime CostFunctionPrime;

        protected OutputLayerBase(int inputs, int outputs, ActivationFunctionType activation, CostFunctionType cost)
            : base(inputs, outputs, activation)
        {
            if (activation == ActivationFunctionType.Softmax && cost != CostFunctionType.LogLikelyhood ||
                cost == CostFunctionType.LogLikelyhood && activation != ActivationFunctionType.Softmax)
                throw new ArgumentException("The softmax activation and log-likelyhood cost function must be used together in a softmax layer");
            (CostFunction, CostFunctionPrime) = CostFunctionProvider.GetCostFunctions(cost);
        }

        /// <summary>
        /// Computes the output delta, with respect to the cost function of the network
        /// </summary>
        /// <param name="yHat">The estimated outputs for the network</param>
        /// <param name="y">The expected outputs for the used inputs</param>
        /// <param name="z">The activity on the output layer</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public float[,] Backpropagate([NotNull] float[,] yHat, [NotNull] float[,] y, [NotNull] float[,] z)
        {
            CostFunctionPrime(yHat, y, z, ActivationFunctions.ActivationPrime);
            return yHat;
        }

        /// <summary>
        /// Calculates the output cost with respect to the cost function currently in use
        /// </summary>
        /// <param name="yHat">The estimated output for the network</param>
        /// <param name="y">The Expected outputs for the inputs used</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public float CalculateCost([NotNull] float[,] yHat, [NotNull] float[,] y) => CostFunction(yHat, y);
    }
}
