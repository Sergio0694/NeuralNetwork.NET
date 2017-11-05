using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="NeuralNetworkNET.SupervisedLearning.BackpropagationNetworkTrainer"/> class and dependencies
    /// </summary>
    [TestClass]
    [TestCategory(nameof(TrainingTest))]
    public class TrainingTest
    {
        [TestMethod]
        public void BatchDivisionTest1()
        {
            Random r = new Random();
            double[,]
                x = r.NextXavierMatrix(60000, 784),
                y = r.NextXavierMatrix(60000, 10);
            IReadOnlyList<TrainingBatch> batches = TrainingBatch.FromDataset(x, y, 1000);
            double[,]
                xs = batches.Select(b => b.X).ToArray().MergeRows(),
                ys = batches.Select(b => b.Y).ToArray().MergeRows();
            Assert.IsTrue(x.ContentEquals(xs));
            Assert.IsTrue(y.ContentEquals(ys));
        }

        [TestMethod]
        public void BatchDivisionTest2()
        {
            Random r = new Random();
            double[,]
                x = r.NextXavierMatrix(20000, 784),
                y = r.NextXavierMatrix(20000, 10);
            IReadOnlyList<TrainingBatch> batches = TrainingBatch.FromDataset(x, y, 333);
            double[,]
                xs = batches.Select(b => b.X).ToArray().MergeRows(),
                ys = batches.Select(b => b.Y).ToArray().MergeRows();
            Assert.IsTrue(x.ContentEquals(xs));
            Assert.IsTrue(y.ContentEquals(ys));
        }
    }
}
