using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.PublicAPIs;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks
{
    /// <summary>
    /// The base class for every neural network implementation
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public abstract class NeuralNetworkBase : INeuralNetwork
    {
        #region Fields and parameters

        /// <summary>
        /// Gets the size of the input layer
        /// </summary>
        [JsonProperty("Inputs", Required = Required.Always)]
        public int InputLayerSize { get; }

        /// <summary>
        /// Gets the size of the output layer
        /// </summary>
        [JsonProperty("Outputs", Required = Required.Always)]
        public int OutputLayerSize { get; }

        /// <summary>
        /// Gets the description of the network hidden layers
        /// </summary>
        [JsonProperty(nameof(HiddenLayers), Required = Required.Always)]
        public abstract IReadOnlyList<int> HiddenLayers { get; }

        /// <summary>
        /// Gets a list of all the weight matrices for the current network
        /// </summary>
        [NotNull, ItemNotNull]
        [JsonProperty(nameof(Weights), Required = Required.Always)]
        protected internal abstract double[][,] Weights { get; }

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
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        /// <remarks>This methods forwards multiple inputs in batch and returns a matrix of results</remarks>
        [PublicAPI]
        [MustUseReturnValue]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public abstract double[,] Forward([NotNull] double[,] input);

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public double CalculateCost(double[] input, double[] y)
        {
            // Forward the input
            double[] yHat = Forward(input);

            // Calculate the cost (half the squared difference)
            double cost = 0;
            for (int i = 0; i < y.Length; i++)
            {
                double
                    delta = y[i] - yHat[i],
                    square = delta * delta;
                cost += square;
            }
            return cost / 2;
        }

        [PublicAPI]
        [Pure]
        public bool Equals([CanBeNull] INeuralNetwork other)
        {
            return other != null &&
                   InputLayerSize == other.InputLayerSize &&
                   OutputLayerSize == other.OutputLayerSize &&
                   HiddenLayers.Count == other.HiddenLayers.Count &&
                   !HiddenLayers.Where((t, i) => t != other.HiddenLayers[i]).Any() &&
                   SerializeWeights().ContentEquals(((NeuralNetworkBase)other).SerializeWeights());
        }

        [PublicAPI]
        [Pure]
        public String SerializeAsJSON() => JsonConvert.SerializeObject(this, Formatting.Indented);

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
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public double CalculateCost([NotNull] double[,] input, [NotNull] double[,] y)
        {
            // Forward the input
            double[,] yHat = Forward(input);

            // Calculate the cost (half the squared difference)
            int h = y.GetLength(0), w = y.GetLength(1);
            double[] v = new double[h];
            bool result = Parallel.For(0, h, i =>
            {
                for (int j = 0; j < w; j++)
                {
                    double
                        delta = y[i, j] - yHat[i, j],
                        square = delta * delta;
                    v[i] += square;
                }
            }).IsCompleted;
            if (!result) throw new Exception("Error while runnig the parallel loop");

            // Sum the partial costs
            double cost = 0;
            for (int i = 0; i < h; i++)
                cost += v[i];
            return cost / 2;
        }

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

        #region Internal methods

        /// <summary>
        /// Serializes the weights of the current instance into a linear array
        /// </summary>
        [PublicAPI]
        [Pure, NotNull]
        internal abstract double[] SerializeWeights();

        /// <summary>
        /// Performs a random crossover with another neural network
        /// </summary>
        /// <param name="other">The other network to use for the crossover</param>
        /// <param name="random">The random instance to use</param>
        [Pure, NotNull]
        internal abstract NeuralNetworkBase Crossover([NotNull] NeuralNetworkBase other, [NotNull] Random random);

        #endregion
    }
}
