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
            NeuralNetwork test = NeuralNetwork.NewRandom(2, 3, 1);
            Random d = new Random();
            var x = d.NextMatrix(3, 2);
            var y = d.NextMatrix(3, 1);
            var gnn = test.ComputeGradient(x, y);

            double[] weights = test.Serialize();
            double[,] 
                w1 = new double[2, 3],
                w2 = new double[3, 1];
            Buffer.BlockCopy(weights, 0, w1, 0, sizeof(double) * w1.Length);
            Buffer.BlockCopy(weights, sizeof(double) * w1.Length, w2, 0, sizeof(double) * w2.Length);
            SingleLayerPerceptron single = new SingleLayerPerceptron(w1, w2);

            var fnn = test.Forward(x);
            var fs = single.Forward(x);

            var gs = single.CostFunctionPrime(x, y);

            Assert.IsTrue(gs.ContentEquals(gnn));
            Assert.IsTrue(fnn.ContentEquals(fs));
        }
    }
}
