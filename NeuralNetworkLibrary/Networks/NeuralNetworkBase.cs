using System;

namespace NeuralNetworkLibrary.Networks
{
    /// <summary>
    /// The base class for every neural network implementation
    /// </summary>
    internal abstract class NeuralNetworkBase : INeuralNetwork
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
        public readonly int HiddenLayerSize;

        /// <summary>
        /// Gets the weights from the inputs to the first hidden layer
        /// </summary>
        protected readonly double[,] W1;

        /// <summary>
        /// Gets the optional threshold for the first layer of hidden neurons
        /// </summary>
        protected readonly double? Z1Threshold;

        /// <summary>
        /// Gets the optional threshold for the second neurons layer
        /// </summary>
        protected readonly double? Z2Threshold;

        /// <summary>
        /// Gets the weights from the first hidden layer
        /// </summary>
        protected readonly double[,] W2;

        #endregion

        /// <summary>
        /// Initializes the readonly fields
        /// </summary>
        /// <param name="input">Input size of the network</param>
        /// <param name="output">Output size of the network</param>
        /// <param name="hiddenSize">The size of the first hidden layer</param>
        /// <param name="w1">The weights from the inputs to the first hidden layer</param>
        /// <param name="w2">The weights from the first hidden layer</param>
        /// <param name="z1Th">Threshold for the hidden neurons layer</param>
        /// <param name="z2Th">Threshold for the second layer of neurons</param>
        protected NeuralNetworkBase(int input, int output, int hiddenSize, double[,] w1, double[,] w2, double? z1Th = null, double? z2Th = null)
        {
            InputLayerSize = input;
            OutputLayerSize = output;
            HiddenLayerSize = hiddenSize;
            W1 = w1;
            W2 = w2;
            Z1Threshold = z1Th;
            Z2Threshold = z2Th;
        }

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        public abstract double[,] Forward(double[,] input);

        /// <summary>
        /// Performs the random crossover with another neural network
        /// </summary>
        /// <param name="other">The other network to use for the crossover</param>
        /// <param name="random">The random instance</param>
        public abstract NeuralNetworkBase Crossover(NeuralNetworkBase other, Random random);
    }
}
