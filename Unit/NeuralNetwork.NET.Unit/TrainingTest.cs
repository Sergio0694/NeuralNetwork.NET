using System;
using System.Collections.Generic;
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
        public void BatchDivisionTest1()
        {
            float[,]
                x = ThreadSafeRandom.NextGlorotNormalMatrix(60000, 784),
                y = ThreadSafeRandom.NextGlorotNormalMatrix(60000, 10);
            BatchesCollection batches = BatchesCollection.FromDataset((x, y), 1000);
            IEnumerable<TrainingBatch> testList = batches.NextEpoch();
            // TODO: check the shuffle is coherent
        }

        [TestMethod]
        public void BatchDivisionTest2()
        {
            float[,]
                x = ThreadSafeRandom.NextGlorotNormalMatrix(20000, 784),
                y = ThreadSafeRandom.NextGlorotNormalMatrix(20000, 10);
            BatchesCollection batches = BatchesCollection.FromDataset((x, y), 333);
            IEnumerable<TrainingBatch> testList = batches.NextEpoch();
            // TODO: check the shuffle is coherent
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
