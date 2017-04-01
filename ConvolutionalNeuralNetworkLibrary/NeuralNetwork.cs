using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        public readonly int HiddenLayerSize;

        /// <summary>
        /// Gets the weights from the inputs to the first hidden layer
        /// </summary>
        protected readonly double[,] W1;

        /// <summary>
        /// Gets the weights from the first hidden layer
        /// </summary>
        protected readonly double[,] W2;

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
        public NeuralNetwork(int input, int output, int hiddenSize, double[,] w1, double[,] w2)
        {
            InputLayerSize = input;
            OutputLayerSize = output;
            HiddenLayerSize = hiddenSize;
            W1 = w1;
            W2 = w2;
        }

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        public double[] Forward(double[] input)
        {
            // TODO
            return null;
        }
    }
}
