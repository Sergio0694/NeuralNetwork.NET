using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="SupervisedLearning.NetworkTrainer"/> class and dependencies
    /// </summary>
    [TestClass]
    [TestCategory(nameof(TrainingTest))]
    public class TrainingTest
    {
        [TestMethod]
        [SuppressMessage("ReSharper", "ReturnValueOfPureMethodIsNotUsed")]
        public void BatchDivisionTest1()
        {
            // Sequential
            float[,]
                x = ThreadSafeRandom.NextGlorotNormalMatrix(60000, 784),
                y = ThreadSafeRandom.NextGlorotNormalMatrix(60000, 10);
            BatchesCollection batches = BatchesCollection.FromDataset((x, y), 1000);
            batches.NextEpoch();
            int xor = 0;
            for (int i = 0; i < 60000; i++)
            {
                float sum = 0;
                for (int j = 0; j < 784; j++) sum += x[i, j];
                for (int j = 0; j < 10; j++) sum += y[i, j];
                xor ^= (int)sum;
            }
            int xorBatch = 0;
            for (int i = 0; i < batches.Count; i++)
            {
                for (int z = 0; z < batches.Batches[i].X.GetLength(0); z++)
                {
                    float sum = 0;
                    for (int j = 0; j < 784; j++) sum += batches.Batches[i].X[z, j];
                    for (int j = 0; j < 10; j++) sum += batches.Batches[i].Y[z, j];
                    xorBatch ^= (int)sum;
                }
            }
            Assert.IsTrue(xor == xorBatch);
        }

        [TestMethod]
        public void BatchDivisionTest2()
        {
            float[][]
                x = Enumerable.Range(0, 60000).Select(_ => ThreadSafeRandom.NextGaussianVector(784)).ToArray(),
                y = Enumerable.Range(0, 60000).Select(_ => ThreadSafeRandom.NextGaussianVector(10)).ToArray();
            BatchesCollection dataset = BatchesCollection.FromDataset(Enumerable.Range(0, 60000).Select(i => (x[i], y[i])), 1000);
            int xor = 0;
            for (int i = 0; i < 60000; i++)
            {
                float sum = 0;
                for (int j = 0; j < 784; j++) sum += x[i][j];
                for (int j = 0; j < 10; j++) sum += y[i][j];
                xor ^= (int)sum;
            }
            int xorBatch = 0;
            for (int i = 0; i < dataset.Count; i++)
            {
                for (int z = 0; z < dataset.Batches[i].X.GetLength(0); z++)
                {
                    float sum = 0;
                    for (int j = 0; j < 784; j++) sum += dataset.Batches[i].X[z, j];
                    for (int j = 0; j < 10; j++) sum += dataset.Batches[i].Y[z, j];
                    xorBatch ^= (int)sum;
                }
            }
            Assert.IsTrue(xor == xorBatch);
        }

        [TestMethod]
        public void BatchInitializationTest()
        {
            float[,]
                x = ThreadSafeRandom.NextUniformMatrix(250, 600, 1000),
                y = ThreadSafeRandom.NextUniformMatrix(250, 10, 500);
            (float[], float[])[] samples = Enumerable.Range(0, 250).Select(i =>
            {
                float[]
                    xv = new float[600],
                    yv = new float[10];
                Buffer.BlockCopy(x, sizeof(float) * i * 600, xv, 0, sizeof(float) * 600);
                Buffer.BlockCopy(y, sizeof(float) * i * 10, yv, 0, sizeof(float) * 10);
                return (xv, yv);
            }).ToArray();
            BatchesCollection
                batch1 = BatchesCollection.FromDataset((x, y), 100),
                batch2 = BatchesCollection.FromDataset(samples, 100);
            Assert.IsTrue(batch1.Batches.Zip(batch2.Batches, (b1, b2) => b1.X.ContentEquals(b2.X) && b1.Y.ContentEquals(b2.Y)).All(b => b));
        }
    }
}
