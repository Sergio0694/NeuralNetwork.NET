using System;
using NeuralNetworkLibrary.Helpers;

namespace NeuralNetworkLibrary.Networks.Implementations
{
    /// <summary>
    /// A neural network with a single hidden neurons layer
    /// </summary>
    internal class NeuralNetwork : NeuralNetworkBase
    {
        /// <summary>
        /// Gets a copy of the internal network weights
        /// </summary>
        public Tuple<double[,], double[,]> Weights => Tuple.Create((double[,])W1.Clone(), (double[,])W2.Clone());

        /// <summary>
        /// Creates a new instance with random weights
        /// </summary>
        /// <param name="input">Input size of the network</param>
        /// <param name="output">Output size of the network</param>
        /// <param name="hidden">The size of the hidden layer</param>
        /// <param name="z1Th">Threshold for the hidden neurons layer</param>
        /// <param name="z2Th">Threshold for the second layer of neurons</param>
        /// <param name="random">The random instance to initialize the weights</param>
        public NeuralNetwork(int input, int output, int hidden, double? z1Th, double? z2Th, Random random) :
            base(input, output, hidden,
            MatrixHelper.RandomMatrix(input, hidden, random),
            MatrixHelper.RandomMatrix(hidden, output, random),
            z1Th, z2Th)
        { }

        /// <summary>
        /// Creates a new instance with the given weights
        /// </summary>
        /// <param name="input">Input size of the network</param>
        /// <param name="output">Output size of the network</param>
        /// <param name="hidden">The size of the hidden layer</param>
        /// <param name="w1">First weights matrix</param>
        /// <param name="w2">Second weights matrix</param>
        /// /// <param name="z1Th">Threshold for the hidden neurons layer</param>
        /// <param name="z2Th">Threshold for the second layer of neurons</param>
        public NeuralNetwork(int input, int output, int hidden, double[,] w1, double[,] w2, double? z1Th, double? z2Th) :
            base(input, output, hidden, w1, w2, z1Th, z2Th)
        { }

        #region Base methods

        // Forwards the input
        public override double[,] Forward(double[,] input)
        {
            // Edge case
            double[,] z2 = MatrixHelper.Multiply(input, W1);
            MatrixHelper.Sigmoid(z2, Z1Threshold);
            double[,] z3 = MatrixHelper.Multiply(z2, W2);
            MatrixHelper.Sigmoid(z3, Z2Threshold);
            return z3;
        }

        // Creates a new instance from another network with the same structure
        public override NeuralNetworkBase Crossover(NeuralNetworkBase other, Random random)
        {
            // Input check
            NeuralNetwork net = other as NeuralNetwork;
            if (net == null) throw new ArgumentException();

            // Crossover
            double[,]
                w1 = MatrixHelper.TwoPointsCrossover(W1, net.W1, random),
                w2 = MatrixHelper.TwoPointsCrossover(W2, net.W2, random);
            return new NeuralNetwork(InputLayerSize, OutputLayerSize, HiddenLayerSize, w1, w2, Z1Threshold, Z2Threshold);
        }

        #endregion
    }
}
