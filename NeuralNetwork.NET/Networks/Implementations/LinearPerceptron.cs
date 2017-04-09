using System;
using System.Collections.Generic;
using System.Text;
using JetBrains.Annotations;

namespace NeuralNetwork.NET.Networks.Implementations
{
    public class LinearPerceptron
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
        /// Gets the description of the network hidden layers
        /// </summary>
        public IReadOnlyList<int> HiddenLayers { get; } = new int[0];

        /// <summary>
        /// Gets the number of neurons in the first hidden layer of the network
        /// </summary>
        protected readonly int H1;

        /// <summary>
        /// Gets the weights from the inputs to the first hidden layer
        /// </summary>
        protected readonly double[,] W1;

        /// <summary>
        /// Gets the weights from the first hidden layer
        /// </summary>
        protected readonly double[,] W2;

        /// <summary>
        /// Gets the transposed W2 weights (used in the gradient calculation)
        /// </summary>
        protected readonly double[,] W2T;

        /// <summary>
        /// Gets the optional threshold for the first layer of hidden neurons
        /// </summary>
        public readonly double? Z1Threshold;

        /// <summary>
        /// Gets the optional threshold for the second neurons layer
        /// </summary>
        public readonly double? Z2Threshold;

        static void Foo()
        {
            
            
        }

        #endregion
    }
}
