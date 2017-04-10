using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A class that represents a neural network with a single hidden layer
    /// </summary>
    public sealed class SingleLayerPerceptron : NeuralNetworkBase
    {
        #region Fields and parameters

        public override IReadOnlyList<int> HiddenLayers { get; }

        /// <summary>
        /// Gets the weights from the inputs to the following layer
        /// </summary>
        [NotNull]
        private readonly double[,] W1;

        /// <summary>
        /// Gets the values in the second layer, before the sigmoid is applied
        /// </summary>
        private double[,] _Z2;

        /// <summary>
        /// Gets the weights from the second layer
        /// </summary>
        [NotNull]
        private readonly double[,] W2;

        /// <summary>
        /// Gets the transposed W2 weights (used in the gradient calculation)
        /// </summary>
        [NotNull]
        private readonly double[,] W2T;

        /// <summary>
        /// Gets the activated values in the second layer
        /// </summary>
        private double[,] _A2;

        /// <summary>
        /// Gets the values in the third layer, before the sigmoid is applied
        /// </summary>
        private double[,] _Z3;

        protected internal override double[][,] Weights => new[] { W1, W2 };

        #endregion

        /// <summary>
        /// Initializes a new instance with the given parameters
        /// </summary>
        /// <param name="w1">The weights from the inputs to the first hidden layer</param>
        /// <param name="w2">The weights from the first hidden layer</param>
        internal SingleLayerPerceptron([NotNull] double[,] w1, [NotNull] double[,] w2) : base(w1.GetLength(0), w2.GetLength(1))
        {
            // Input check
            if (w1.GetLength(1) != w2.GetLength(0))
                throw new ArgumentOutOfRangeException("The size of the inputs isn't correct");

            // Parameters setup
            HiddenLayers = new[] { w1.GetLength(1) };
            W1 = w1;
            W2 = w2;
            W2T = W2.Transpose();
        }

        /// <summary>
        /// Creates a new random instance with the given number of inputs and outputs
        /// </summary>
        /// <param name="inputs">The number of input nodes</param>
        /// <param name="size">The number of nodes in the hidden layer</param>
        /// <param name="outputs">The number of output nodes</param>
        [NotNull]
        internal static SingleLayerPerceptron NewRandom(int inputs, int size, int outputs)
        {
            Random random = new Random();
            double[,]
                w1 = random.NextMatrix(inputs, size),
                w2 = random.NextMatrix(size, outputs);
            return new SingleLayerPerceptron(w1, w2);
        }

        #region Single processing

        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public override double[] Forward(double[] input)
        {
            double[]
                z2 = input.Multiply(W1), // Input >> hidden layer
                a2 = z2.Sigmoid(), // Hidden layer activation
                z3 = a2.Multiply(W2), // Hidden >> output layer
                yHat = z3.Sigmoid(); // Output layer activation
            return yHat;
        }

        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal override double[] CostFunctionPrime(double[] input, double[] y)
        {
            // Locally forward the input to hold a references to the intermediate values
            double[]
                z2 = input.Multiply(W1), // Input >> hidden layer
                a2 = z2.Sigmoid(), // Hidden layer activation
                z3 = a2.Multiply(W2), // Hidden >> output layer
                yHat = z3.Sigmoid(); // Output layer activation

            // Calculate the negative delta for later use
            double[] negativeDelta = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
                negativeDelta[i] = -(y[i] - yHat[i]);

            // Derivative with respect to W2
            double[]
                z3prime = z3.SigmoidPrime(),
                delta3 = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
                delta3[i] = negativeDelta[i] * z3prime[i];
            double[] dJdW2 = new double[_A2.Length];
            for (int i = 0; i < _A2.Length; i++)
                dJdW2[i] = a2[i] * delta3[i];

            // Derivative with respect to W1
            double[]
                delta3w2t = delta3.Multiply(W2T),
                z2prime = z2.SigmoidPrime(),
                delta2 = new double[delta3w2t.Length];
            for (int i = 0; i < delta2.Length; i++)
                delta2[i] = delta3w2t[i] * z2prime[i];
            double[] dJdW1 = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
                dJdW1[i] = input[i] * delta2[i];

            // Return the results
            return dJdW1.Concat(dJdW2).ToArray();
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
            _A2 = _Z2.Sigmoid();
            _Z3 = _A2.Multiply(W2);

            // Hidden layer >> output (with activation)
            double[,] yHat = _Z3.Sigmoid();
            return yHat;
        }

        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal override double[] CostFunctionPrime(double[,] input, double[,] y)
        {
            // Forward the input
            double[,] yHat = Forward(input);

            // Calculate the negative delta for later use
            int h = y.GetLength(0), w = y.GetLength(1);
            double[,] negativeDelta = new double[h, w];
            bool result = ParallelCompatibilityWrapper.Instance.Invoke(0, h, i =>
            {
                unsafe
                {
                    // Fix the pointers
                    fixed (double* nd = negativeDelta, py = y, pyHat = yHat)
                    {
                        for (int j = 0; j < w; j++)
                            nd[i * w + j] = -(py[i * w + j] - pyHat[i * w + j]);
                    }
                }
            });
            if (!result) throw new Exception("Error while runnig the parallel loop");

            // Derivative with respect to W2
            double[,]
                z3prime = _Z3.SigmoidPrime(),
                delta3 = new double[h, w];
            result = ParallelCompatibilityWrapper.Instance.Invoke(0, h, i =>
            {
                unsafe
                {
                    // Fix the pointers
                    fixed (double* d3 = delta3, nd = negativeDelta, z3p = z3prime)
                    {
                        for (int j = 0; j < w; j++)
                            d3[i * w + j] = nd[i * w + j] * z3p[i * w + j];
                    }
                }
            });
            if (!result) throw new Exception("Error while runnig the parallel loop");
            double[,]
                a2t = _A2.Transpose(),
                dJdW2 = a2t.Multiply(delta3);

            // Derivative with respect to W1
            double[,]
                delta3w2t = delta3.Multiply(W2T),
                z2prime = _Z2.SigmoidPrime();
            int
                delta3w2th = delta3w2t.GetLength(0),
                delta3w2tw = delta3w2t.GetLength(1);
            double[,] delta2 = new double[delta3w2th, delta3w2tw];
            for (int i = 0; i < delta3w2th; i++)
            for (int j = 0; j < delta3w2tw; j++)
                delta2[i, j] = delta3w2t[i, j] * z2prime[i, j];
            double[,]
                xt = input.Transpose(),
                dJdW1 = xt.Multiply(delta2);

            // Return the results
            return dJdW1.Cast<double>().Concat(dJdW2.Cast<double>()).ToArray();
        }

        #endregion

        #region Tools

        /// <summary>
        /// Deserializes a neural network from the input weights and parameters
        /// </summary>
        /// <param name="inputs">The number of input nodes</param>
        /// <param name="size">The number of nodes in the hidden layer</param>
        /// <param name="outputs">The number of output nodes</param>
        /// <param name="w1w2">The serialized network weights</param>
        [PublicAPI]
        [Pure, NotNull]
        internal static SingleLayerPerceptron Deserialize(int inputs, int size, int outputs, [NotNull] double[] w1w2)
        {
            // Checks
            if (inputs <= 0 || size <= 0 || outputs <= 0 || w1w2.Length < inputs * size + size * outputs)
                throw new ArgumentOutOfRangeException("The inputs are invalid");

            // Parse the data
            double[,]
                w1 = new double[inputs, size],
                w2 = new double[size, outputs];
            int w1length = sizeof(double) * w1.Length;
            Buffer.BlockCopy(w1w2, 0, w1, 0, w1length);
            Buffer.BlockCopy(w1w2, w1length, w2, 0, sizeof(double) * w2.Length);

            // Create the new network to use
            return new SingleLayerPerceptron(w1, w2);
        }

        [PublicAPI]
        [Pure]
        internal override double[] SerializeWeights() => W1.Cast<double>().Concat(W2.Cast<double>()).ToArray();

        // Creates a new instance from another network with the same structure
        [Pure]
        internal override NeuralNetworkBase Crossover(NeuralNetworkBase other, Random random)
        {
            // Input check
            SingleLayerPerceptron net = other as SingleLayerPerceptron;
            if (net == null) throw new ArgumentException();

            // Crossover
            double[,]
                w1 = random.TwoPointsCrossover(W1, net.W1),
                w2 = random.TwoPointsCrossover(W2, net.W2);
            return new SingleLayerPerceptron(w1, w2);
        }

        #endregion
    }
}
