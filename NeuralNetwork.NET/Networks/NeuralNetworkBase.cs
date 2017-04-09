using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.PublicAPIs;

namespace NeuralNetworkNET.Networks
{
    /// <summary>
    /// The base class for every neural network implementation
    /// </summary>
    public abstract class NeuralNetworkBase : INeuralNetwork
    {
        #region Fields and parameters

        /// <summary>
        /// Gets the size of the input layer
        /// </summary>
        public int InputLayerSize { get; }

        /// <summary>
        /// Gets the size of the output layer
        /// </summary>
        public int OutputLayerSize { get; }

        /// <summary>
        /// Gets the description of the network hidden layers
        /// </summary>
        public abstract IReadOnlyList<int> HiddenLayers { get; }

        #endregion

        /// <summary>
        /// Initializes the readonly fields
        /// </summary>
        /// <param name="inputs">The number of input nodes</param>
        /// <param name="outputs">The number of output nodes</param>
        protected internal NeuralNetworkBase(int inputs, int outputs)
        {
            if (inputs <= 0 || outputs <= 0) throw new ArgumentOutOfRangeException("The inputs and outputs must be positive numbers");
            InputLayerSize = inputs;
            OutputLayerSize = outputs;
        }

        #region Interface

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        /// <remarks>This methods processes a single input row and outputs a single result</remarks>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public abstract double[] Forward(double[] input);

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public abstract double CalculateCost(double[] input, double[] y);

        #endregion

        #region Training methods

        /// <summary>
        /// Computes the derivative with respect to W1 and W2 for a given input and result
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal abstract double[] CostFunctionPrime([NotNull] double[] input, [NotNull] double[] y);

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        /// <remarks>This methods forwards multiple inputs in batch and returns a matrix of results</remarks>
        [PublicAPI]
        [MustUseReturnValue]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal abstract double[,] Forward([NotNull] double[,] input);

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal abstract double CalculateCost([NotNull] double[,] input, [NotNull] double[,] y);

        /// <summary>
        /// Computes the derivative with respect to W1 and W2 for a given input and result
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal abstract double[] CostFunctionPrime([NotNull] double[,] input, [NotNull] double[,] y);

        #endregion

        /// <summary>
        /// Performs the random crossover with another neural network
        /// </summary>
        /// <param name="other">The other network to use for the crossover</param>
        /// <param name="random">The random instance</param>
        //public abstract NeuralNetworkBase Crossover(NeuralNetworkBase other, Random random);

        /// <summary>
        /// Serializes the weights of the current instance into a linear array
        /// </summary>
        [PublicAPI]
        [Pure]
        [NotNull]
        internal abstract double[] SerializeWeights();

        //protected abstract IReadOnlyList<XNode> SerializeToXML();

        //
    }
}
