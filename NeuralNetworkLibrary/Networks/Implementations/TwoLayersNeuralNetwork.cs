using System;
using System.IO;
using NeuralNetworkLibrary.Helpers;

namespace NeuralNetworkLibrary.Networks.Implementations
{
    internal class TwoLayersNeuralNetwork : NeuralNetwork
    {
        #region Fields and parameters

        /// <summary>
        /// Gets the size of the second hidden layer
        /// </summary>
        public readonly int SecondHiddenLayerSize;

        /// <summary>
        /// Gets the weights of the second hidden layer
        /// </summary>
        private readonly double[,] W3;

        /// <summary>
        /// Gets the optional threshold for the last layer of hidden neurons
        /// </summary>
        public readonly double? Z3Threshold;

        /// <summary>
        /// Gets a copy of the internal weights of the network
        /// </summary>
        public new Tuple<double[,], double[,], double[,]> Weights => Tuple.Create((double[,])W1.Clone(), (double[,])W2.Clone(), (double[,])W3.Clone());

        #endregion

        /// <summary>
        /// Creates a new instance with random weights
        /// </summary>
        /// <param name="input">Input size of the network</param>
        /// <param name="output">Output size of the network</param>
        /// <param name="firstHidden">The size of the first hidden layer</param>
        /// <param name="secondHidden">The size of the second hidden layer</param>
        /// <param name="z1Th">Threshold for the hidden neurons layer</param>
        /// <param name="z2Th">Threshold for the second layer of neurons</param>
        /// <param name="z3Th">Threshold for the last layers of neurons</param>
        /// <param name="random">The random instance to initialize the weights</param>
        public TwoLayersNeuralNetwork(int input, int output, int firstHidden, int secondHidden, double? z1Th, double? z2Th, double? z3Th, Random random) :
            base(input, output, firstHidden,
                MatrixHelper.RandomMatrix(input, firstHidden, random),
                MatrixHelper.RandomMatrix(firstHidden, secondHidden, random),
                z1Th, z2Th)
        {
            SecondHiddenLayerSize = secondHidden;
            W3 = MatrixHelper.RandomMatrix(secondHidden, output, random);
            Z3Threshold = z3Th;
        }

        /// <summary>
        /// Creates a new instance with the given weights
        /// </summary>
        /// <param name="input">Input size of the network</param>
        /// <param name="output">Output size of the network</param>
        /// <param name="firstHidden">The size of the first hidden layer</param>
        /// <param name="secondHidden">The size of the second hidden layer</param>
        /// <param name="w1">First weights matrix</param>
        /// <param name="w2">Second weights matrix</param>
        /// <param name="w3">Third weights matrix</param>
        /// <param name="z1Th">Threshold for the hidden neurons layer</param>
        /// <param name="z2Th">Threshold for the second layer of neurons</param>
        /// <param name="z3Th">Threshold for the last layers of neurons</param>
        public TwoLayersNeuralNetwork(int input, int output, int firstHidden, int secondHidden, 
            double[,] w1, double[,] w2, double[,] w3,
            double? z1Th, double? z2Th, double? z3Th) :
            base(input, output, firstHidden, w1, w2, z1Th, z2Th)
        {
            SecondHiddenLayerSize = secondHidden;
            W3 = w3;
            Z3Threshold = z3Th;
        }

        #region Base methods

        // Forwards the input
        public override double[,] Forward(double[,] input)
        {
            double[,]
                z3 = base.Forward(input),
                z4 = MatrixHelper.Multiply(z3, W3);
            MatrixHelper.Sigmoid(z4, Z3Threshold);
            return z3;
        }

        // Creates a new instance from another network with the same structure
        public override NeuralNetworkBase Crossover(NeuralNetworkBase other, Random random)
        {
            // Input check
            TwoLayersNeuralNetwork net = other as TwoLayersNeuralNetwork;
            if (net == null) throw new ArgumentException();

            // Crossover
            double[,]
                w1 = MatrixHelper.TwoPointsCrossover(W1, net.W1, random),
                w2 = MatrixHelper.TwoPointsCrossover(W2, net.W2, random),
                w3 = MatrixHelper.TwoPointsCrossover(W3, net.W3, random);
            return new TwoLayersNeuralNetwork(InputLayerSize, OutputLayerSize, HiddenLayerSize, SecondHiddenLayerSize, 
                w1, w2, w3, Z1Threshold, Z2Threshold, Z3Threshold);
        }

        #endregion

        #region Serialization

        // Serializes the instance
        public override byte[] Serialize()
        {
            // Variables initialization
            const int fixedSize = 12;
            byte[] single = base.Serialize();
            int size = single.Length + fixedSize + SecondHiddenLayerSize * OutputLayerSize * 8;
            byte[] buffer = new byte[size];

            // Create the stream and serialize the instance
            using (MemoryStream stream = new MemoryStream(buffer))
            {
                stream.Write(single, 0, single.Length);
                stream.Write(BitConverter.GetBytes(SecondHiddenLayerSize), 0, 4);
                stream.Write(BitConverter.GetBytes(Z3Threshold ?? double.MinValue), 0, 8);
                W3.ForEach((i, j) => stream.Write(BitConverter.GetBytes(W3[i, j]), 0, 8));
            }
            return buffer;
        }

        #endregion
    }
}
