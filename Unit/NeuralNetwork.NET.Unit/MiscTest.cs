using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Settings;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for various helper methods and extensions
    /// </summary>
    [TestClass]
    [TestCategory(nameof(MiscTest))]
    public class MiscTest
    {
        [TestMethod]
        public void Partition()
        {
            IEnumerable<int> items = Enumerable.Range(0, 25);
            StringBuilder builder = new StringBuilder();
            foreach (var chunk in items.Partition(7))
                builder.Append($"{chunk.Aggregate(String.Empty, (s, i) => $"{s} {i}")}\n");
            String result = builder.ToString();
            Assert.IsTrue(result.Equals(" 0 1 2 3 4 5 6\n 7 8 9 10 11 12 13\n 14 15 16 17 18 19 20\n 21 22 23 24\n"));
        }

        [TestMethod]
        public void Fill()
        {
            float[] v = new float[127];
            v.AsSpan().Fill(() => 1.8f);
            Assert.IsTrue(v.All(f => f.EqualsWithDelta(1.8f)));
        }

        [TestMethod]
        public void CudaSupport()
        {
            Assert.IsFalse(CuDnnNetworkLayers.IsCudaSupportAvailable);
        }

        [TestMethod]
        public void ThresholdAccuracyTest()
        {
            AccuracyTester tester = AccuracyTesters.Threshold();
            Span<float>
                yHat = new[] { 0.1f, 0.4f, 0.6f, 0.99f, 0.1f, 0.73f },
                y1 = new[] { 0.8f, 0.1f, 0.8f, 0.3f, 0.2f, 0.66f },
                y2 = new[] { 0.5f, 0.3f, 0.8f, 1, 0.004f, 0.990f };
            Assert.IsFalse(tester(yHat, y1));
            Assert.IsTrue(tester(yHat, y2));
        }

        [TestMethod]
        public void DistanceAccuracyTest()
        {
            AccuracyTester tester = AccuracyTesters.Distance(0.4f);
            Span<float>
                yHat = new[] { 0.1f, 0.4f, 0.6f, 0.99f, 0.1f, 0.73f },
                y1 = new[] { 0.3f, 0.5f, 0.93f, 1.2f, 0.2f, 0.66f },
                y2 = new[] { 0.55f, 0.3f, 0.8f, 1, 0.004f, 0.990f };
            Assert.IsTrue(tester(yHat, y1));
            Assert.IsFalse(tester(yHat, y2));
        }

        [TestMethod]
        public void TrimVerbatim()
        {
            const String text = @"import matplotlib.pyplot as plt
                                  x = [$VALUES$]
                                  plt.grid(linestyle=""dashed"")
                                  plt.ylabel(""$YLABEL$"")
                                  plt.xlabel(""Epoch"")
                                  plt.plot(x)
                                  plt.show()";
            String[] lines =
            {
                "import matplotlib.pyplot as plt",
                "x = [$VALUES$]",
                "plt.grid(linestyle=\"dashed\")",
                "plt.ylabel(\"$YLABEL$\")",
                "plt.xlabel(\"Epoch\")",
                "plt.plot(x)",
                "plt.show()"
            };
            String expected = lines.Skip(1).Aggregate(lines[0], (s, l) => $"{s}{Environment.NewLine}{l}") + Environment.NewLine;
            Assert.IsTrue(text.TrimVerbatim().Equals(expected));
        }

        [TestMethod]
        public void ConvolutionInfoFactory()
        {
            ConvolutionInfo info = ConvolutionInfo.Same()(TensorInfo.Image<Alpha8>(28, 28), (3, 3));
            Assert.IsTrue(info.VerticalPadding == 1 && info.HorizontalPadding == 1);
            info = ConvolutionInfo.Same()(TensorInfo.Image<Alpha8>(28, 28), (5, 5));
            Assert.IsTrue(info.VerticalPadding == 2 && info.HorizontalPadding == 2);
            info = ConvolutionInfo.Same(ConvolutionMode.Convolution, 2, 2)(TensorInfo.Image<Alpha8>(10, 10), (3, 3));
            Assert.IsTrue(info.VerticalPadding == 6 && info.HorizontalPadding == 6);
        }
    }
}
