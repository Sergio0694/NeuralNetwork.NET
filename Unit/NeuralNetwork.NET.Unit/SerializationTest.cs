using System;
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
                Assert.IsTrue(stream.TryRead(out PoolingInfo copy));
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
                Assert.IsTrue(stream.TryRead(out PoolingMode copy));
                Assert.IsTrue(mode == copy);
            }
        }

        [TestMethod]
        public void StreamSerialize()
        {
            using (MemoryStream stream = new MemoryStream())
            {
                float[] w = WeightsProvider.NewFullyConnectedWeights(784, 30, WeightsInitializationMode.GlorotNormal);
                stream.WriteShuffled(w);
                Assert.IsTrue(stream.Position == sizeof(float) * w.Length);
                stream.Seek(0, SeekOrigin.Begin);
                float[] t = stream.ReadUnshuffled(w.Length);
                Assert.IsTrue(w.ContentEquals(t));
            }
        }
    }
}
