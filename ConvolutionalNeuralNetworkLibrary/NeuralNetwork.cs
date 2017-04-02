using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace ConvolutionalNeuralNetworkLibrary
{
    public class NeuralNetwork
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
        private double[] _Z2;

        /// <summary>
        /// Gets the hidden layer activated values
        /// </summary>
        private double[] _A2;

        /// <summary>
        /// Gets the latest output layer values, before the sigmoid is applied
        /// </summary>
        private double[] _Z3;

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

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        [MustUseReturnValue]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public double[] Forward([NotNull] double[] input)
        {
            // Input >> hidden layer (with activation)
            _Z2 = MatrixHelper.Multiply(input, W1);
            _A2 = MatrixHelper.Sigmoid(_Z2);
            _Z3 = MatrixHelper.Multiply(_A2, W2);

            // Hidden layer >> output (with activation)
            double[] yHat = MatrixHelper.Sigmoid(_Z3);
            return yHat;
        }

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
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
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public (double[], double[]) CostFunctionPrime([NotNull] double[] input, [NotNull] double[] y)
        {
            // Forward the input
            double[] yHat = Forward(input);

            // Calculate the negative delta for later use
            double[] negativeDelta = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
                negativeDelta[i] = -(y[i] - yHat[i]);

            // Derivative with respect to W2
            double[]
                z3prime = MatrixHelper.SigmoidPrime(_Z3),
                delta3 = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
                delta3[3] = negativeDelta[i] * z3prime[i];
            double[] dJdW2 = new double[_A2.Length];
            for (int i = 0; i < _A2.Length; i++)
                dJdW2[i] = _A2[i] * delta3[i];

            // Derivative with respect to W1
            double[]
                delta3w2t = MatrixHelper.Multiply(delta3, W2T),
                z2prime = MatrixHelper.SigmoidPrime(_Z2),
                delta2 = new double[delta3w2t.Length];
            for (int i = 0; i < delta2.Length; i++)
                delta2[i] = delta3w2t[i] * z2prime[i];
            double[] dJdW1 = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
                dJdW1[i] = input[i] * delta2[i];

            // Return the results
            return (dJdW1, dJdW2);
        }

        /// <summary>
        /// Serializes the current weights into a linear array of (W1.h*W1.w) + (W2.h*W2.w) elements
        /// </summary>
        [Pure]
        [NotNull]
        public double[] SerializeWeights() => W1.Cast<double>().Concat(W2.Cast<double>()).ToArray();
    }
}
