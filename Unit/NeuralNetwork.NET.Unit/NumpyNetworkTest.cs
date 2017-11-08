using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Implementations;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// A class with some test methods for the posted network class from python
    /// </summary>
    [TestClass]
    [TestCategory(nameof(NumpyNetworkTest))]
    public class NumpyNetworkTest
    {
        [TestMethod]
        public void TestNumpy1()
        {
            NumpyNetwork network = new NumpyNetwork(1, 2, 1)
            {
                weights =
                {
                    [0] = new[,] { { 1.34856747 }, { -1.16546082 } },
                    [1] = new[,] { { -0.73764399, -0.69019199 } }
                },
                biases =
                {
                    [0] = new[,] { { 0.45206544 }, { 0.66440039 } },
                    [1] = new[,] { { -0.01439235 } }
                }
            };
            double[,] value = network.feedforward(new[,] { { 1d } });
            Assert.IsTrue((value[0, 0] - 0.28743771).Abs() < 0.1);
        }

        [TestMethod]
        public void TestNumpy2()
        {
            NumpyNetwork network = new NumpyNetwork(1, 2, 1)
            {
                weights =
                {
                    [0] = new[,] { { 1.34856747 }, { -1.16546082 } },
                    [1] = new[,] { { -0.73764399, -0.69019199 } }
                },
                biases =
                {
                    [0] = new[,] { { 0.45206544 }, { 0.66440039 } },
                    [1] = new[,] { { -0.01439235 } }
                }
            };
            (double[][,] dJdb, double[][,] dJdw) = network.backprop(new[,] { { 1.2 } }, new[,] { { 1.2 } });
            Assert.IsTrue(dJdb.Length == 2 &&
                          dJdb[0].GetLength(0) == 2 && dJdb[0].GetLength(1) == 1 &&
                          dJdb[1].Length == 1);
            Assert.IsTrue(dJdw.Length == 2 &&
                          dJdw[0].GetLength(0) == 2 && dJdw[0].GetLength(1) == 1 &&
                          dJdw[1].GetLength(0) == 1 && dJdw[1].GetLength(1) == 2);
            Assert.IsTrue(dJdb[0][0, 0].EqualsWithDelta(0.01375305, 1e-5) &&
                          dJdb[0][1, 0].EqualsWithDelta(0.02834903, 1e-5) &&
                          dJdb[1][0, 0].EqualsWithDelta(-0.18744699, 1e-5) &&
                          dJdw[0][0, 0].EqualsWithDelta(0.01650366, 1e-5) &&
                          dJdw[0][1, 0].EqualsWithDelta(0.03401884, 1e-5) &&
                          dJdw[1][0, 0].EqualsWithDelta(-0.16645057, 1e-5) &&
                          dJdw[1][0, 1].EqualsWithDelta(-0.06078609, 1e-5));
        }
    }
}
