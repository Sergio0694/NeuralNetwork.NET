using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Implementations;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// A test class for the gradient methods
    /// </summary>
    [TestClass]
    [TestCategory(nameof(GradientTests))]
    public class GradientTests
    {
        [TestMethod]
        public void GradientTest1()
        {
            SingleLayerPerceptron single = SingleLayerPerceptron.NewRandom(2, 2, 2);
            var gradient = single.CostFunctionPrime(new[,] { { 1.0, 2.0 } }, new[,] { { 3.0, 4.0 } });
            SingleLayerPerceptron second = SingleLayerPerceptron.NewRandom(3, 2, 2);
            var gradient2 = second.CostFunctionPrime(new[,] { { 7.0, 1.0, 2.0 } }, new[,] { { 3.0, 4.0 } });
            var gradient2_1 = second.CostFunctionPrime(new[,] { { 7.0, 1.0, 2.0 }, {1,2,3} }, new[,] { { 3.0, 4.0 },{0.6, 0.7} });

            NeuralNetwork test = NeuralNetwork.NewRandom(2, 3, 2, 1);
            Random d = new Random();
            var g = test.ComputeGradient(d.NextMatrix(3, 2), d.NextMatrix(3, 1));

            NeuralNetwork dnn = NeuralNetwork.NewRandom(3, 2, 2);
            var gdnn = dnn.ComputeGradient(new[,] { { 1.0, 2.0, 4.2 } }, new[,] { { 3.0, 4.0 } });
        }
    }
}
