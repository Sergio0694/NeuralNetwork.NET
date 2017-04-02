using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace ConvolutionalNeuralNetworkLibrary
{
    /// <summary>
    /// A class that represents a neural network with a single hidden layer
    /// </summary>
    public sealed class NeuralNetwork
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
        /// Gets the number of neurons in the first hidden layer of the network
        /// </summary>
        public int HiddenLayerSize { get; }

        /// <summary>
        /// Gets the weights from the inputs to the first hidden layer
        /// </summary>
        private readonly double[,] W1;

        /// <summary>
        /// Gets the weights from the first hidden layer
        /// </summary>
        private readonly double[,] W2;

        /// <summary>
        /// Gets the transposed W2 weights (used in the gradient calculation)
        /// </summary>
        private readonly double[,] W2T;

        /// <summary>
        /// Gets the latest hidden layer values, before the sigmoid is applied
        /// </summary>
        private double[,] _Z2;

        /// <summary>
        /// Gets the hidden layer activated values
        /// </summary>
        private double[,] _A2;

        /// <summary>
        /// Gets the latest output layer values, before the sigmoid is applied
        /// </summary>
        private double[,] _Z3;

        #endregion

        /// <summary>
        /// Initializes a new instance with the given parameters
        /// </summary>
        /// <param name="input">Input size of the network</param>
        /// <param name="output">Output size of the network</param>
        /// <param name="hiddenSize">The size of the first hidden layer</param>
        /// <param name="w1">The weights from the inputs to the first hidden layer</param>
        /// <param name="w2">The weights from the first hidden layer</param>
        /// <param name="z1Th">Threshold for the hidden neurons layer</param>
        /// <param name="z2Th">Threshold for the second layer of neurons</param>
        public NeuralNetwork(int input, int output, int hiddenSize, 
            [NotNull] double[,] w1, [NotNull] double[,] w2)
        {
            if (input != w1.GetLength(0) ||
                w1.GetLength(1) != hiddenSize ||
                hiddenSize != w2.GetLength(0) ||
                w2.GetLength(1) != output)
            {
                throw new ArgumentOutOfRangeException("The size of the inputs isn't correct");
            }
            InputLayerSize = input;
            OutputLayerSize = output;
            HiddenLayerSize = hiddenSize;
            W1 = w1;
            W2 = w2;
            W2T = MatrixHelper.Transpose(W2);
        }

        #region Single processing

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        /// <remarks>This methods processes a single input row and outputs a single result</remarks>
        [PublicAPI]
        [MustUseReturnValue]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public double[] Forward([NotNull] double[] input)
        {
            double[] 
                z2 = MatrixHelper.Multiply(input, W1), // Input >> hidden layer
                a2 = MatrixHelper.Sigmoid(z2), // Hidden layer activation
                z3 = MatrixHelper.Multiply(a2, W2), // Hidden >> output layer
                yHat = MatrixHelper.Sigmoid(z3); // Output layer activation
            return yHat;
        }

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public double CalculateCost([NotNull] double[] input, [NotNull] double[] y)
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

        /// <summary>
        /// Computes the derivative with respect to W1 and W2 for a given input and result
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public (double[], double[]) CostFunctionPrime([NotNull] double[] input, [NotNull] double[] y)
        {
            // Locally forward the input to hold a references to the intermediate values
            double[]
                z2 = MatrixHelper.Multiply(input, W1), // Input >> hidden layer
                a2 = MatrixHelper.Sigmoid(z2), // Hidden layer activation
                z3 = MatrixHelper.Multiply(a2, W2), // Hidden >> output layer
                yHat = MatrixHelper.Sigmoid(z3); // Output layer activation

            // Calculate the negative delta for later use
            double[] negativeDelta = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
                negativeDelta[i] = -(y[i] - yHat[i]);

            // Derivative with respect to W2
            double[]
                z3prime = MatrixHelper.SigmoidPrime(z3),
                delta3 = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
                delta3[i] = negativeDelta[i] * z3prime[i];
            double[] dJdW2 = new double[_A2.Length];
            for (int i = 0; i < _A2.Length; i++)
                dJdW2[i] = a2[i] * delta3[i];

            // Derivative with respect to W1
            double[]
                delta3w2t = MatrixHelper.Multiply(delta3, W2T),
                z2prime = MatrixHelper.SigmoidPrime(z2),
                delta2 = new double[delta3w2t.Length];
            for (int i = 0; i < delta2.Length; i++)
                delta2[i] = delta3w2t[i] * z2prime[i];
            double[] dJdW1 = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
                dJdW1[i] = input[i] * delta2[i];

            // Return the results
            return (dJdW1, dJdW2);
        }

        #endregion

        #region Batch processing

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        /// <remarks>This methods forwards multiple inputs in batch and returns a matrix of results</remarks>
        [PublicAPI]
        [MustUseReturnValue]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public double[,] Forward([NotNull] double[,] input)
        {
            // Perform the batch processing
            _Z2 = MatrixHelper.Multiply(input, W1);
            _A2 = MatrixHelper.Sigmoid(_Z2);
            _Z3 = MatrixHelper.Multiply(_A2, W2);

            // Hidden layer >> output (with activation)
            double[,] yHat = MatrixHelper.Sigmoid(_Z3);
            return yHat;
        }

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public double CalculateCost([NotNull] double[,] input, [NotNull] double[,] y)
        {
            // Forward the input
            double[,] yHat = Forward(input);

            // Calculate the cost (half the squared difference)
            double cost = 0, h = y.GetLength(0), w = y.GetLength(1);
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    double
                        delta = y[i, j] - yHat[i, j],
                        square = delta * delta;
                    cost += square;
                }
            }
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
        public (double[,], double[,]) CostFunctionPrime([NotNull] double[,] input, [NotNull] double[,] y)
        {
            // Forward the input
            double[,] yHat = Forward(input);

            // Calculate the negative delta for later use
            int h = y.GetLength(0), w = y.GetLength(1);
            double[,] negativeDelta = new double[h, w];
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    negativeDelta[i, j] = -(y[i, j] - yHat[i, j]);

            // Derivative with respect to W2
            double[,]
                z3prime = MatrixHelper.SigmoidPrime(_Z3),
                delta3 = new double[h, w];
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    delta3[i, j] = negativeDelta[i, j] * z3prime[i, j];
            double[,]
                a2t = MatrixHelper.Transpose(_A2),
                dJdW2 = MatrixHelper.Multiply(a2t, delta3);

            // Derivative with respect to W1
            double[,]
                delta3w2t = MatrixHelper.Multiply(delta3, W2T),
                z2prime = MatrixHelper.SigmoidPrime(_Z2);
            int
                delta3w2th = delta3w2t.GetLength(0),
                delta3w2tw = delta3w2t.GetLength(1);
            double[,] delta2 = new double[delta3w2th, delta3w2tw];
            for (int i = 0; i < delta3w2th; i++)
                for (int j = 0; j < delta3w2tw; j++)
                    delta2[i, j] = delta3w2t[i, j] * z2prime[i, j];
            double[,]
                xt = MatrixHelper.Transpose(input),
                dJdW1 = MatrixHelper.Multiply(xt, delta2);

            // Return the results
            return (dJdW1, dJdW2);
        }

        #endregion

        /// <summary>
        /// Serializes the current weights into a linear array of (W1.h*W1.w) + (W2.h*W2.w) elements
        /// </summary>
        [Pure]
        [NotNull]
        internal double[] SerializeWeights() => W1.Cast<double>().Concat(W2.Cast<double>()).ToArray();
    }
}
