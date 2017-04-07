using System;
using System.Linq;
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
            W2T = W2.Transpose();
        }

        #region Single processing

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        /// <remarks>This methods processes a single input row and outputs a single result</remarks>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public double[] Forward([NotNull] double[] input)
        {
            double[]
                z2 = input.Multiply(W1), // Input >> hidden layer
                a2 = z2.Sigmoid(), // Hidden layer activation
                z3 = a2.Multiply(W2), // Hidden >> output layer
                yHat = z3.Sigmoid(); // Output layer activation
            return yHat;
        }

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [PublicAPI]
        [Pure]
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
        [CollectionAccess(CollectionAccessType.Read)]
        public (double[], double[]) CostFunctionPrime([NotNull] double[] input, [NotNull] double[] y)
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
            _Z2 = input.Multiply(W1);
            _A2 = _Z2.Sigmoid();
            _Z3 = _A2.Multiply(W2);

            // Hidden layer >> output (with activation)
            double[,] yHat = _Z3.Sigmoid();
            return yHat;
        }

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
            ParallelLoopResult result = Parallel.For(0, h, i =>
            {
                for (int j = 0; j < w; j++)
                {
                    double
                        delta = y[i, j] - yHat[i, j],
                        square = delta * delta;
                    v[i] += square;
                }
            });
            if (!result.IsCompleted) throw new Exception("Error while runnig the parallel loop");

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
        [CollectionAccess(CollectionAccessType.Read)]
        public (double[,], double[,]) CostFunctionPrime([NotNull] double[,] input, [NotNull] double[,] y)
        {
            // Forward the input
            double[,] yHat = Forward(input);

            // Calculate the negative delta for later use
            int h = y.GetLength(0), w = y.GetLength(1);
            double[,] negativeDelta = new double[h, w];
            ParallelLoopResult result = Parallel.For(0, h, i =>
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
            if (!result.IsCompleted) throw new Exception("Error while runnig the parallel loop");

            // Derivative with respect to W2
            double[,]
                z3prime = _Z3.SigmoidPrime(),
                delta3 = new double[h, w];
            result = Parallel.For(0, h, i =>
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
            if (!result.IsCompleted) throw new Exception("Error while runnig the parallel loop");
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
            return (dJdW1, dJdW2);
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
        [Pure]
        [NotNull]
        public static NeuralNetwork Deserialize(int inputs, int size, int outputs, double[] w1w2)
        {
            double[,]
                    w1 = new double[inputs, size],
                    w2 = new double[size, outputs];
            int w1length = sizeof(double) * w1.Length;
            Buffer.BlockCopy(w1w2, 0, w1, 0, w1length);
            Buffer.BlockCopy(w1w2, w1length, w2, 0, sizeof(double) * w2.Length);

            // Create the new network to use
            return new NeuralNetwork(inputs, outputs, size, w1, w2);
        }

        /// <summary>
        /// Serializes the current weights into a linear array of (W1.h*W1.w) + (W2.h*W2.w) elements
        /// </summary>
        [PublicAPI]
        [Pure]
        [NotNull]
        public double[] SerializeWeights() => W1.Cast<double>().Concat(W2.Cast<double>()).ToArray();

        #endregion
    }
}
