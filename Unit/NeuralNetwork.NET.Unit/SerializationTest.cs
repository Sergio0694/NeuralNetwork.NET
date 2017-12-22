using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the serialization methods
    /// </summary>
    [TestClass]
    [TestCategory(nameof(SerializationTest))]
    public class SerializationTest
    {
        [TestMethod]
        public void StructSerialize()
        {
            PoolingInfo info = PoolingInfo.New(PoolingMode.AverageIncludingPadding, 3, 3, 1, 1, 2, 2);
            using (MemoryStream stream = new MemoryStream())
            {
                stream.Write(info);
                stream.Seek(0, SeekOrigin.Begin);
                PoolingInfo copy = stream.Read<PoolingInfo>();
                Assert.IsTrue(info.Equals(copy));
            }
        }

        [TestMethod]
        public void EnumSerialize()
        {
            PoolingMode mode = PoolingMode.AverageIncludingPadding;
            using (MemoryStream stream = new MemoryStream())
            {
                stream.Write(mode);
                stream.Seek(0, SeekOrigin.Begin);
                PoolingMode copy = stream.Read<PoolingMode>();
                Assert.IsTrue(mode == copy);
            }
        }

        [TestMethod]
        public void StreamSerialize()
        {
            using (MemoryStream stream = new MemoryStream())
            {
                float[,] m = WeightsProvider.NewFullyConnectedWeights(784, 30, WeightsInitializationMode.GlorotNormal).AsMatrix(784, 30);
                stream.Write(m);
                byte[] test = new byte[10];
                stream.Seek(-10, SeekOrigin.Current);
                stream.Read(test, 0, 10);
                Assert.IsTrue(test.Any(b => b != 0));
                Assert.IsTrue(stream.Position == sizeof(float) * m.Length);
                stream.Seek(0, SeekOrigin.Begin);
                float[,] copy = stream.ReadFloatArray(784, 30);
                Assert.IsTrue(m.ContentEquals(copy));
            }
            using (MemoryStream stream = new MemoryStream())
            {
                float[] v = WeightsProvider.NewBiases(723, BiasInitializationMode.Gaussian);
                stream.Write(v);
                byte[] test = new byte[10];
                stream.Seek(-10, SeekOrigin.Current);
                stream.Read(test, 0, 10);
                Assert.IsTrue(test.Any(b => b != 0));
                Assert.IsTrue(stream.Position == sizeof(float) * v.Length);
                stream.Seek(0, SeekOrigin.Begin);
                float[] copy = stream.ReadFloatArray(723);
                Assert.IsTrue(v.ContentEquals(copy));
            }
        }
    }
}
