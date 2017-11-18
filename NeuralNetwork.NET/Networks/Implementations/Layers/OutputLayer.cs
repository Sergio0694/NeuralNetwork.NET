using System;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// An output layer with a variable cost function
    /// </summary>
    internal class OutputLayer : OutputLayerBase
    {
        public OutputLayer(int inputs, int outputs, ActivationFunctionType activation, CostFunctionType cost)
            : base(inputs, outputs, activation, cost)
        {
            if (activation == ActivationFunctionType.Softmax)
                throw new ArgumentException("The softmax activation can only be used in a softmax layer");
        }
    }
}
