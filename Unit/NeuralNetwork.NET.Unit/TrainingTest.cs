using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="NeuralNetworkNET.SupervisedLearning.NetworkTrainer"/> class and dependencies
    /// </summary>
    [TestClass]
    [TestCategory(nameof(TrainingTest))]
    public class TrainingTest
    {
        [TestMethod]
        public void BatchDivisionTest1()
        {
            Random r = new Random();
            float[,]
                x = r.NextXavierMatrix(60000, 784),
                y = r.NextXavierMatrix(60000, 10);
            BatchesCollection batches = BatchesCollection.FromDataset((x, y), 1000);
            IEnumerable<TrainingBatch> testList = batches.NextEpoch();
            // TODO: check the shuffle is coherent
        }

        [TestMethod]
        public void BatchDivisionTest2()
        {
            Random r = new Random();
            float[,]
                x = r.NextXavierMatrix(20000, 784),
                y = r.NextXavierMatrix(20000, 10);
            BatchesCollection batches = BatchesCollection.FromDataset((x, y), 333);
            IEnumerable<TrainingBatch> testList = batches.NextEpoch();
            // TODO: check the shuffle is coherent
        }

        [TestMethod]
        public void BatchInitializationTest()
        {
            Random r = new Random();
            float[,]
                x = r.NextGaussianMatrix(250, 600),
                y = r.NextGaussianMatrix(250, 10);
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
