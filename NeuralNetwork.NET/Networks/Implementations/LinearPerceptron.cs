using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Networks.Implementations
{
    public class LinearPerceptron : NeuralNetworkBase
    {
        #region Fields and parameters

        /// <summary>
        /// Gets the description of the network hidden layers
        /// </summary>
        public override IReadOnlyList<int> HiddenLayers { get; } = new int[0];

        /// <summary>
        /// Gets the weights from the inputs to the following layer
        /// </summary>
        [NotNull]
        protected readonly double[,] W1;

        /// <summary>
        /// Gets the values in the second layer, before the sigmoid is applied
        /// </summary>
        protected double[,] _Z2;

        protected internal override double[][,] Weights => new[] { W1 };

        #endregion

        /// <summary>
        /// Creates a new instance with the given values
        /// </summary>
        /// <param name="inputs">The number of inputs</param>
        /// <param name="outputs">The number of output nodes</param>
        /// <param name="w1">The first weights in the perceptron</param>
        internal LinearPerceptron(int inputs, int outputs, [NotNull] double[,] w1) : base(inputs, outputs)
        {
            if (w1.Length == 0) throw new ArgumentOutOfRangeException("The weights can't be empty");
            if (w1.GetLength(0) != inputs) throw new ArgumentOutOfRangeException("The size of the weights matrix isn't valid");
            W1 = w1;
        }

        /// <summary>
        /// Creates a new random instance with the given number of inputs and outputs
        /// </summary>
        /// <param name="inputs">The number of input nodes</param>
        /// <param name="outputs">The number of output nodes</param>
        [NotNull]
        internal static LinearPerceptron NewRandom(int inputs, int outputs)
        {
            return new LinearPerceptron(inputs, outputs, new Random().NextMatrix(inputs, outputs));
        }

        #region Single processing

        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public override double[] Forward(double[] input)
        {
            double[]
                z2 = input.Multiply(W1), // Input >> output layer
                yHat = z2.Sigmoid(); // Output layer activation
            return yHat;
        }

        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal override double[] CostFunctionPrime(double[] input, double[] y)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Batch processing

        [PublicAPI]
        [MustUseReturnValue]
        [CollectionAccess(CollectionAccessType.Read)]
        internal override double[,] Forward(double[,] input)
        {
            // Perform the batch processing
            _Z2 = input.Multiply(W1);

            // Hidden layer >> output (with activation)
            double[,] yHat = _Z2.Sigmoid();
            return yHat;
        }

        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal override double[] CostFunctionPrime(double[,] input, double[,] y)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Tools

        /// <summary>
        /// Deserializes a linear perceptron from the input weights and parameters
        /// </summary>
        /// <param name="inputs">The number of input nodes</param>
        /// <param name="outputs">The number of output nodes</param>
        /// <param name="weights">The serialized network weights</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        internal static LinearPerceptron Deserialize(int inputs, int outputs, [NotNull] double[] weights)
        {
            // Checks
            if (inputs <= 0 || outputs <= 0 || weights.Length < inputs * outputs)
                throw new ArgumentOutOfRangeException("The inputs are invalid");

            // Parse the data
            double[,]
                w1 = new double[inputs, outputs];
            int w1length = sizeof(double) * w1.Length;
            Buffer.BlockCopy(weights, 0, w1, 0, w1length);

            // Create the new network to use
            return new LinearPerceptron(inputs, outputs, w1);
        }

        [PublicAPI]
        [Pure]
        internal override double[] SerializeWeights() => W1.Cast<double>().ToArray();

        // Creates a new instance from another network with the same structure
        [Pure]
        internal override NeuralNetworkBase Crossover(NeuralNetworkBase other, Random random)
        {
            // Input check
            LinearPerceptron net = other as LinearPerceptron;
            if (net == null) throw new ArgumentException();

            // Crossover
            double[,] w1 = random.TwoPointsCrossover(W1, net.W1);
            return new LinearPerceptron(InputLayerSize, OutputLayerSize, w1);
        }

        #endregion
    }
}
