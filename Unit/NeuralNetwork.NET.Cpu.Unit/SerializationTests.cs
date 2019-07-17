using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs.Info;

namespace NeuralNetwork.NET.Cpu.Unit
{
    /// <summary>
    /// Test class for the serialization methods
    /// </summary>
    [TestClass]
    [TestCategory(nameof(SerializationTest))]
    public class SerializationTest
    {
        [TestMethod]
        public void FloatSerialize()
        {
            var value = 24343.1341f;
            using (MemoryStream stream = new MemoryStream())
            {
                stream.Write(value);
                stream.Seek(0, SeekOrigin.Begin);
                Assert.IsTrue(stream.TryRead(out float copy));
                Assert.IsTrue(Math.Abs(value - copy) < 0.0001f);
            }
        }

        [TestMethod]
        public void EnumSerialize()
        {
            var mode = PoolingMode.AverageIncludingPadding;
            using (MemoryStream stream = new MemoryStream())
            {
                stream.Write(mode);
                stream.Seek(0, SeekOrigin.Begin);
                Assert.IsTrue(stream.TryRead(out PoolingMode copy));
                Assert.IsTrue(mode == copy);
            }
        }

        [TestMethod]
        public void StructSerialize()
        {
            var info = ConvolutionInfo.New(ConvolutionMode.CrossCorrelation, 34, 22, 12, 11);
            using (MemoryStream stream = new MemoryStream())
            {
                stream.Write(info);
                stream.Seek(0, SeekOrigin.Begin);
                Assert.IsTrue(stream.TryRead(out ConvolutionInfo copy));
                Assert.IsTrue(info.Equals(copy));
            }
        }

        [TestMethod]
        public void StreamSerialize1()
        {
            var data = new[] { 7.77f };

            using (MemoryStream stream = new MemoryStream())
            {
                stream.Write(data.AsSpan());
                Assert.IsTrue(stream.Position == sizeof(float));
                stream.Seek(0, SeekOrigin.Begin);

                var copy = stream.TryRead<float>(1);
                Assert.IsTrue(data.AsSpan().ContentEquals(copy));
            }
        }

        [TestMethod]
        public void StreamSerialize2()
        {
            using (var data = Tensor.New(4, 3, 32, 32))
            {
                data.Span.Fill(() => ConcurrentRandom.Instance.NextFloat());

                using (MemoryStream stream = new MemoryStream())
                {
                    stream.Write(data.Span);
                    Assert.IsTrue(stream.Position == sizeof(float) * data.Shape.NCHW);
                    stream.Seek(0, SeekOrigin.Begin);

                    var copy = stream.TryRead<float>(data.Shape.NCHW);
                    Assert.IsTrue(data.Span.ContentEquals(copy));
                }
            }
        }
    }
}
