using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A multilayer deep neural network with an arbitrary number of hidden layers
    /// </summary>
    public sealed class MultilayerPerceptron : NeuralNetworkBase
    {
        #region Parameters

        public override IReadOnlyList<int> HiddenLayers { get; }

        protected internal override double[][,] Weights { get; }

        #endregion

        /// <summary>
        /// Creates a new multilayer perceptron with the input weights for each layer
        /// </summary>
        /// <param name="layers"></param>
        internal MultilayerPerceptron([NotNull] IReadOnlyList<double[,]> layers) :
            base(layers.Count >= 3 ? layers[0].GetLength(0) : throw new ArgumentOutOfRangeException("A multilayer perceptron must have at least two hidden layers"),
                layers[layers.Count - 1].GetLength(1))
        {
            // Checks
            if (layers.Any(n => n.Length == 0))
                throw new ArgumentOutOfRangeException("The weights layers can't be empty");
            for (int i = 0; i < layers.Count - 1; i++)
                if (layers[i].GetLength(1) != layers[i + 1].GetLength(0))
                    throw new ArgumentOutOfRangeException("Invalid weights");

            // Setup
            Weights = layers.ToArray();
            HiddenLayers = layers.Take(layers.Count - 1).Select(l => l.GetLength(1)).ToArray();
        }

        /// <summary>
        /// Creates a new random instance with the given number of inputs and outputs
        /// </summary>
        /// <param name="inputs">The number of input nodes</param>
        /// <param name="outputs">The number of output nodes</param>
        /// <param name="layers">The number of nodes in each hidden layer</param>
        [NotNull]
        internal static MultilayerPerceptron NewRandom(int inputs, int outputs, [NotNull] IReadOnlyList<int> layers)
        {
            throw new NotImplementedException();
        }

        #region Single processing

        // Execute the multiplication + sigmoid activation for each layer
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public override double[] Forward(double[] input)
        {
            return Weights.Aggregate(input, (value, layer) => value.Multiply(layer).Sigmoid());
        }

        internal override double[] CostFunctionPrime(double[] input, double[] y)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Batch processing

        public override double[,] Forward(double[,] input)
        {
            throw new NotImplementedException();
        }

        internal override double[] CostFunctionPrime(double[,] input, double[,] y)
        {
            throw new NotImplementedException();
        }

        #endregion

        /// <summary>
        /// Deserializes a neural network from the input weights and parameters
        /// </summary>
        /// <param name="layersInfo">The height and width of each network layer</param>
        /// <param name="weights">The serialized network weights</param>
        [PublicAPI]
        [Pure, NotNull]
        internal static MultilayerPerceptron Deserialize([NotNull] IReadOnlyList<(int, int)> layersInfo, [NotNull] double[] weights)
        {
            throw new NotImplementedException();
        }

        [PublicAPI]
        [Pure]
        internal override double[] SerializeWeights() => Weights.Flatten();

        // Creates a new instance from another network with the same structure
        [Pure]
        internal override NeuralNetworkBase Crossover(NeuralNetworkBase other, Random random)
        {
            throw new NotImplementedException();
        }
    }
}
